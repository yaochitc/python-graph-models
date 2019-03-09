import tensorflow as tf


class Trainer(object):
    def __init__(self,
                 loss,
                 learning_rate=0.001):
        self.loss = loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.step = self.optimizer.minimize(self.loss)
