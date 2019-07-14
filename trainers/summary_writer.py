import tensorflow as tf
import os


class SummaryWriter(object):

    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = None

    def add_summary(self, sess, data, itr):
        if self.writer is None:
            self.writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        self.writer.add_summary(data, itr)
