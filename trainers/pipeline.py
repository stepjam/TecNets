import tensorflow as tf


class Pipeline(object):

    def __init__(self, consumers, learning_rate=1e-4, grad_clip=None,
                 saver=None, loader=None):
        self.consumers = consumers
        self.grad_clip = grad_clip
        self.opt = tf.train.AdamOptimizer(learning_rate)
        self.total_loss = None
        self.sess = None
        self.gradients = None
        self.saver = saver
        self.loader = loader
        self.got_outputs = False

    def get_session(self):
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        return self.sess

    def get_outputs(self):
        outputs = {}
        for consumer in self.consumers:
            outputs = consumer.consume(outputs)
        self.got_outputs = True
        return outputs

    def load(self):
        if self.loader is None:
            return 1
        if not self.got_outputs:
            raise RuntimeError(
                'get_outputs() needs to be called before loading a model.')
        return self.loader.load(self.get_session())

    def save(self, itr):
        if self.saver is not None:
            self.saver.save(self.get_session(), itr)

    def get_summaries(self, prefix):
        if self.total_loss is None:
            self.get_loss()
        summaries = [tf.summary.scalar(prefix + '_total_loss', self.total_loss)]
        for consumer in self.consumers:
            summaries.append(consumer.get_summaries(prefix))

        if self.gradients is None:
            raise RuntimeError('Call get_train_op before this.')
        for grad, var in self.gradients:
            summaries.append(tf.summary.histogram(var.name, var))
            summaries.append(tf.summary.histogram(var.name + '/gradient', grad))

        return tf.summary.merge(summaries)

    def get_loss(self):
        loss = 0
        for consumer in self.consumers:
            loss += consumer.get_loss()
        self.total_loss = loss
        return loss

    def get_train_op(self, loss):
        # gvs = self.opt.compute_gradients(loss)
        gradients = tf.gradients(loss, tf.trainable_variables())
        self.gradients = list(zip(gradients, tf.trainable_variables()))
        if self.grad_clip is not None:
            self.gradients = [
                (tf.clip_by_value(grad, -self.grad_clip, self.grad_clip)
                 if grad is not None else grad, var)
                for grad, var in self.gradients]
        return self.opt.apply_gradients(self.gradients)
