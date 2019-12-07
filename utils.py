from __future__ import print_function

import tensorflow.compat.v1 as tf
import numpy as np

def KL_divergence_between_normal(mu1, sigma1, mu2, sigma2):
    """
    Assume tf.tensor do element-wise operation as expected
    """
    return tf.reduce_sum(tf.log(tf.math.truediv(sigma2, sigma1)) + \
            tf.math.truediv(sigma1*sigma1 + (mu1-mu2), 2*sigma2*sigma2) - \
            0.5)


def deviation_rho(rho):
    return tf.sqrt(tf.log(1 + tf.exp(rho)))
if __name__ == "__main__":
    mu1 = np.array([[1, 0],[0, 1]], dtype=np.float32)
    mu2 = np.array([[0,1],[1,0]], dtype=np.float32)
    sigma1 = np.array([[1,1],[1,1]], dtype=np.float32)
    sigma2 = np.array([[1,1],[1,1]], dtype=np.float32)

    loss = KL_divergence_between_normal(mu1, sigma1, mu2, sigma2)
    with tf.Session() as sess:
        print(loss)
