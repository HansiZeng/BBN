from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime

import tensorflow.compat.v1 as tf
# Helper libraries
import struct
import numpy as np
#import matplotlib.pyplot as plt

import utils

print(tf.__version__)


"""
helper function
"""
def log_gaussian(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi) - tf.log(tf.abs(sigma)) - (x - mu) ** 2 / (2 * sigma ** 2) + 1e-7

def gaussian(x, mu, sigma):
    return tf.exp(log_gaussian(x, mu, sigma)) + 1e-7

def log_spike_and_slab_prior(x, sigma1, sigma2, pi):
    #print(sigma1, sigma2, pi)
    #return tf.log(gaussian(x, 0.0, 0.36)) 
    return tf.log(pi * gaussian(x, 0.0, sigma1)+ (1-pi)*gaussian(x, 0.0, sigma2))

def get_random(shape, avg, std):
    return tf.random_normal(shape, mean=avg, stddev=std)

# build determined W first
class MnistClassification():
    def __init__(self, input_size, hidden_size, learning_rate, net_struct, forward_only=True, batch_num=6):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.net_struct = net_struct
        self.forward_only = forward_only
        self.print_ops = []
        self.global_step = tf.Variable(0, trainable=False)
        self.batch_num = batch_num
        # build
        self._build_placeholder()
        self._build_graph_and_get_loss()

        self.saver = tf.train.Saver(tf.global_variables())

    def _build_placeholder(self):
        self.images = tf.placeholder(dtype=tf.float32, shape=(None, self.input_size), name="images")
        self.labels = tf.placeholder(dtype=tf.int32, shape=(None), name="labels")

    def _build_graph_and_get_loss(self):
        # get variables
        if "random" not in self.net_struct:
            self.W1 = tf.Variable(tf.random_normal([self.input_size, self.hidden_size], stddev=0.35), name="W1")
            self.W2 = tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size], stddev=0.35), name="W2")
            self.W3 = tf.Variable(tf.random_normal([self.hidden_size, 10], stddev=0.35), name="W3")
            self.b1 = tf.Variable(tf.zeros([self.hidden_size]))
            self.b2 = tf.Variable(tf.zeros([self.hidden_size]))
            self.b3 = tf.Variable(tf.zeros([10]))
        else:
            self.mu_1 = tf.Variable(tf.random_normal([self.input_size, self.hidden_size], stddev=0.35), name="mu1")
            self.rho_1 = tf.Variable(tf.random_normal([self.input_size, self.hidden_size], stddev=0.35), name="rho1")
            self.bmu_1 = tf.Variable(tf.zeros([self.hidden_size]))
            self.brho_1 = tf.Variable(tf.zeros([self.hidden_size]))

            self.mu_2 = tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size], stddev=0.35), name="mu_2")
            self.rho_2 = tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size], stddev=0.35), name="rho_2")
            self.bmu_2 = tf.Variable(tf.zeros([self.hidden_size]))
            self.brho_2 = tf.Variable(tf.zeros([self.hidden_size]))

            self.mu_3 = tf.Variable(tf.random_normal([self.hidden_size, 10], stddev=0.35), name="mu_3")
            self.rho_3 = tf.Variable(tf.random_normal([self.hidden_size, 10], stddev=0.35), name="rho_3")
            self.bmu_3 = tf.Variable(tf.zeros([10]))
            self.brho_3 = tf.Variable(tf.zeros([10]))

            # build graph
            self.W1 = self.mu_1 + get_random((self.input_size, self.hidden_size),0.,.01) * tf.math.sqrt(tf.math.log(1 + tf.math.exp(self.rho_1)))
            self.b1 = self.bmu_1 + get_random((self.hidden_size,),0.,.01) * tf.math.sqrt(tf.math.log(1 + tf.math.exp(self.brho_1)))

            self.W2 = self.mu_2 + get_random((self.hidden_size, self.hidden_size),0.,.01) * tf.math.sqrt(tf.math.log(1 + tf.math.exp(self.rho_2)))
            self.b2 = self.bmu_2 + get_random((self.hidden_size,),0., .01) * tf.math.sqrt(tf.math.log(1 + tf.math.exp(self.brho_2)))

            self.W3 = self.mu_3 + get_random((self.hidden_size, 10),0.,.01) * tf.math.sqrt(tf.math.log(1 + tf.math.exp(self.rho_3)))
            self.b3 = self.bmu_3 + get_random((10,),0., .01) * tf.math.sqrt(tf.math.log(1 + tf.math.exp(self.brho_3)))

        self.tmp = None
        self.y = None
        # build graph
        self.tmp = tf.matmul(self.images, self.W1) + self.b1
        self.tmp = tf.nn.relu(self.tmp, name="relu1")

        self.tmp = tf.matmul(self.tmp, self.W2) + self.b2
        self.tmp = tf.nn.relu(self.tmp, name="relu2")

        self.y = tf.matmul(self.tmp, self.W3) + self.b3
        self.y = tf.nn.softmax(self.y, axis=1) + 1e-7

        # create one hot encoding for labels
        self.one_hot_labels = tf.one_hot(self.labels, depth=10)
        #self.print_ops.append(tf.print("labels: ", self.labels, tf.shape(self.labels)))
        #self.print_ops.append(tf.print("One hot labels: ", self.one_hot_labels, tf.shape(self.one_hot_labels)))

        # compute loss
        self.loss = tf.reduce_sum(-tf.math.log(self.y) * self.one_hot_labels)
        if "random"  in self.net_struct:
            log_pw, log_qw = 0.0, 0.0
            if "spik_and_slab" not in self.net_struct:
                print("run standard prior!!!")
                log_pw += tf.reduce_sum(log_gaussian(self.W1, 0.0, 1.0))
                log_pw += tf.reduce_sum(log_gaussian(self.b1, 0.0, 1.0))
                log_pw += tf.reduce_sum(log_gaussian(self.W2, 0.0, 1.0))
                log_pw += tf.reduce_sum(log_gaussian(self.b2, 0.0, 1.0))
                log_pw += tf.reduce_sum(log_gaussian(self.W3, 0.0, 1.0))
                log_pw += tf.reduce_sum(log_gaussian(self.b3, 0.0, 1.0))
            else:
                print("run spik!!!")
                #self.print_ops.append(tf.print("simga 1", 1/tf.exp(1.0)))
                log_pw += tf.reduce_sum(log_spike_and_slab_prior(self.W1, 1.0/tf.exp(1.0), 1.0/tf.exp(6.0), 0.25))
                log_pw += tf.reduce_sum(log_spike_and_slab_prior(self.b1, 1/tf.exp(1.0), 1.0/tf.exp(6.0), 0.25))
                log_pw += tf.reduce_sum(log_spike_and_slab_prior(self.W2, 1.0/tf.exp(1.0), 1.0/tf.exp(6.0), 0.25))
                log_pw += tf.reduce_sum(log_spike_and_slab_prior(self.b2, 1.0/tf.exp(1.0), 1.0/tf.exp(6.0), 0.25))
                log_pw += tf.reduce_sum(log_spike_and_slab_prior(self.W3, 1.0/tf.exp(1.0), 1.0/tf.exp(6.0), 0.25))
                log_pw += tf.reduce_sum(log_spike_and_slab_prior(self.b3, 1.0/tf.exp(1.0), 1.0/tf.exp(6.0), 0.25))

            log_qw += tf.reduce_sum(log_gaussian(self.W1, self.mu_1, tf.math.sqrt(tf.math.log(1 + tf.math.exp(self.rho_1)))))
            log_qw += tf.reduce_sum(log_gaussian(self.b1, self.bmu_1, tf.math.sqrt(tf.math.log(1 + tf.math.exp(self.brho_1)))))
            log_qw += tf.reduce_sum(log_gaussian(self.W2, self.mu_2, tf.math.sqrt(tf.math.log(1 + tf.math.exp(self.rho_2)))))
            log_qw += tf.reduce_sum(log_gaussian(self.b2, self.bmu_2, tf.math.sqrt(tf.math.log(1 + tf.math.exp(self.brho_2)))))
            log_qw += tf.reduce_sum(log_gaussian(self.W3, self.mu_3, tf.math.sqrt(tf.math.log(1 + tf.math.exp(self.rho_3)))))
            log_qw += tf.reduce_sum(log_gaussian(self.b3, self.bmu_3, tf.math.sqrt(tf.math.log(1 + tf.math.exp(self.brho_3)))))
            print(self.batch_num, "!!!!!!!!")
            #self.print_ops.append(tf.print("prior loss: ", log_qw-log_pw))
            #self.loss += (log_qw - log_pw) / self.batch_num
        #self.print_ops.append(tf.print("mask: ", -tf.math.log(self.y) * self.one_hot_labels))
        #self.print_ops.append(tf.print("the step loss: ", self.loss))

        # check whether the model can overfit the train batch
        self.preds = tf.cast(tf.argmax(self.y, axis=1), dtype=tf.int32)
        self.accu = tf.reduce_sum(tf.cast(tf.equal(self.preds, self.labels), dtype=tf.float32)) / \
                                    tf.cast(tf.shape(self.preds)[0], tf.float32)
        #self.print_ops.append(tf.print("preds: ", self.preds, tf.shape(self.preds)))
        #self.print_ops.append(tf.print("labels: ", self.labels, tf.shape(self.labels)))

        if not self.forward_only:
            # apply gradients
            self.update = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,global_step=self.global_step)
        else:
            pass



    def step(self, session, input_feed, forward_only):
        if not forward_only:
            output_feed = [self.loss, self.accu, self.update, self.print_ops]
        else:
            output_feed = [self.accu]

        outputs = session.run(output_feed, input_feed)

        if not forward_only:
            return outputs[0], outputs[1]
        else:
            return outputs[0]


class Dataset():
    def __init__(self, model, image_path, label_path, batch_size):
        self.model = model
        self.image_path = image_path
        self.label_path = label_path
        self.images, self.labels = self._read_images_and_labels()
        self.images = self.images / 126.0
        self.batch_size = batch_size

    def _read_images_and_labels(self):
        with open(self.image_path,'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            print("label size: ", size)

            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            data = data.reshape((size, nrows*ncols))

        with open(self.label_path, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            print("label size: ", size)
            labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))

        return data, labels

    def initilize_epoch(self):
        self.cur_idx = 0
        pertumation_idxs = np.random.permutation(self.images.shape[0])
        self.images = self.images[pertumation_idxs, :]
        self.labels = self.labels[pertumation_idxs]

    def get_train_batch(self):
        input_feed = {}

        if self.cur_idx + self.batch_size > self.images.shape[0]:
            has_next = False
            input_feed[self.model.images.name] = self.images[self.cur_idx: self.images.shape[0]]
            input_feed[self.model.labels.name] = self.labels[self.cur_idx: self.images.shape[0]]

            return input_feed, has_next
        else:
            has_next = True
            input_feed[self.model.images.name] = self.images[self.cur_idx: self.cur_idx+self.batch_size]
            input_feed[self.model.labels.name] = self.labels[self.cur_idx: self.cur_idx+self.batch_size]

            self.cur_idx += self.batch_size
            return input_feed, has_next


    def get_test_batch(self):
        input_feed = {}

        input_feed[self.model.images.name] = self.images[0: self.images.shape[0]]
        input_feed[self.model.labels.name] = self.labels[0: self.images.shape[0]]

        return input_feed

def train(args):

    # place for all hyperparamters and settings
    ckpt_file = ""
    epochs = 1000
    learning_rate = 1e-3
    batch_size = 1000
    input_size = 28*28
    hidden_size = 200
    num_batch = 60000 / float(batch_size) 
    output = ""



    # tune
    for hidden_size in [400, 800, 1200]:
        for learning_rate in [1e-3]:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                # train file
                image_path = "train-images-idx3-ubyte"
                label_path = "train-labels-idx1-ubyte"
                model = MnistClassification(input_size, hidden_size, learning_rate, net_struct=args.net_struct, forward_only=False, batch_num=num_batch)
                dataset = Dataset(model, image_path, label_path, batch_size)
                init_op = tf.initialize_all_variables()
                sess.run(init_op)
                for i in range(epochs):
                    print("In epoch: ", i)
                    has_next = True
                    idx = 0

                    dataset.initilize_epoch()
                    while has_next:
                        idx+=1

                        input_feed, has_next = dataset.get_train_batch()
                        #print(input_feed.keys())

                        loss, accu = model.step(sess, input_feed, forward_only=False)

                        if idx % 10 == 0:
                            print("loss: %.3f\t accuracy: %.3f "%(loss/batch_size, accu))

        #ckpt_path = "./" + "mnist_det_weight.ckpt"
        #model.saver.save(sess, ckpt_path, global_step=model.global_step)


                # test file
                image_path = "t10k-images-idx3-ubyte"
                label_path = "t10k-labels-idx1-ubyte"
                dataset = Dataset(model, image_path, label_path, batch_size)
                input_feed = dataset.get_test_batch()
                accu = model.step(sess, input_feed, forward_only=True)
                print("accuracy: ", accu)
                output += str(hidden_size) + "_" + str(learning_rate) + "_" + str(args.net_struct) + "_" + str(num_batch)+ "\t accu:" + str(accu) + '\n'
                print(output)
                output += ""
    with open("record.txt"+str(datetime.datetime.now()), 'w') as fout:
        fout.write(output)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_struct")
    return parser.parse_args() 

if __name__ == "__main__":
    args = _parse_args()
    train(args)
