from consumers.consumer import Consumer
import tensorflow as tf


class MarginLoss(Consumer):

    def __init__(self, margin, loss_lambda=1.0):
        self.margin = margin
        self.loss_lambda = loss_lambda
        self.loss_embedding = None
        self.embedding_accuracy = None

    def _norm(self, vecs, axis=1):
        mag = tf.sqrt(tf.reduce_sum(tf.square(vecs), axis=axis, keep_dims=True))
        return vecs / tf.maximum(mag, 1e-6)

    def consume(self, inputs):

        # (batch, sup, emb_size)
        semb = self.get(inputs, 'support_embedding')
        qemb = self.get(inputs, 'query_embedding')
        qemb_shape = tf.shape(qemb)
        batch_size, query_size = qemb_shape[0], qemb_shape[1]
        qemb = tf.reshape(qemb, (batch_size * query_size, -1))

        # Shape (batch, embsize)
        support_sentences = self._norm(
            tf.reduce_mean(self._norm(semb, axis=2), axis=1), axis=1)
        inputs['sentences'] = support_sentences

        # Similarities of every sentence with every query
        # Shape (batch, batch * queries)
        similarities = tf.matmul(support_sentences, qemb, transpose_b=True)
        # Shape (batch, batch, queries)
        similarities = tf.reshape(similarities,
                                  (batch_size, batch_size, query_size))

        # Gets the diagonal to give (batch, query)
        positives = tf.boolean_mask(similarities, tf.eye(batch_size))
        positives_ex = tf.expand_dims(positives, axis=1)  # (batch, 1, query)

        negatives = tf.boolean_mask(similarities,
                                    tf.equal(tf.eye(batch_size), 0))
        # (batch, batch-1, query)
        negatives = tf.reshape(negatives, (batch_size, batch_size - 1, -1))

        loss = tf.maximum(0.0, self.margin - positives_ex + negatives)
        loss = tf.reduce_mean(loss)

        self.loss_embedding = self.loss_lambda * loss

        # Summaries
        max_of_negs = tf.reduce_max(negatives, axis=1)  # (batch, query)
        accuracy = tf.greater(positives, max_of_negs)
        self.embedding_accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
        inputs['loss_embedding'] = self.loss_embedding
        inputs['embedding_accuracy'] = self.embedding_accuracy
        return inputs

    def get_summaries(self, prefix):
        return [
            tf.summary.scalar(prefix + 'embedding_accuracy',
                              self.verify(self.embedding_accuracy)),
            tf.summary.scalar(prefix + 'loss_embedding',
                              self.verify(self.loss_embedding))
        ]

    def get_loss(self):
        return self.verify(self.loss_embedding)
