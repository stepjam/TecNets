from consumers.consumer import Consumer
import tensorflow as tf
from networks.input_output import *


class ImitationLoss(Consumer):

    def __init__(self, support_lambda=1.0, query_lambda=1.0):
        self.support_lambda = support_lambda
        self.query_lambda = query_lambda
        self.loss_support = None
        self.loss_query = None

    def consume(self, inputs):

        # (batch, 2, actions)
        a = self.get(inputs, 'output_actions')
        labels = self.get(inputs, 'ctrnet_outputs')

        support_loss = tf.losses.mean_squared_error(a[:, 0], labels[:, 0])
        query_loss = tf.losses.mean_squared_error(a[:, 1], labels[:, 1])

        self.loss_support = self.support_lambda * support_loss
        self.loss_query = self.query_lambda * query_loss
        inputs['loss_support'] = self.loss_support
        inputs['loss_query'] = self.loss_query
        return inputs

    def get_summaries(self, prefix):
        return [
            tf.summary.scalar(
                prefix + '_support_loss', self.verify(self.loss_support)),
            tf.summary.scalar(
                prefix + '_query_loss', self.verify(self.loss_query))
        ]

    def get_loss(self):
        return self.verify(self.loss_support) + self.verify(self.loss_query)
