import tensorflow as tf
import time

VAL_SUMMARY_INTERVAL = 100
SUMMARY_INTERVAL = 100
SAVE_INTERVAL = 10000
EVAL_INTERVAL = 25000


class ILTrainer(object):

    def __init__(self, pipeline, outputs, generator, iterations,
                 summary_writer=None, eval=None):
        self.pipeline = pipeline
        self.generator = generator
        self.outputs = outputs
        self.iterations = iterations
        self.summary_writer = summary_writer
        self.eval = eval

        if eval is not None:
            # Convenience for plotting eval successes in tensorboard
            self.eval_summary_in = tf.placeholder(tf.float32)
            self.eval_summary = tf.summary.scalar('evaluation_success',
                                                  self.eval_summary_in)

    def train(self):

        sess = self.pipeline.get_session()
        train_handle, validation_handle = self.generator.get_handles(sess)

        outputs = self.outputs
        total_loss = self.pipeline.get_loss()
        train_op = self.pipeline.get_train_op(total_loss)
        train_summaries = self.pipeline.get_summaries('train')
        validation_summaries = self.pipeline.get_summaries('validation')

        tf.global_variables_initializer().run()

        # Load if we have supplied a checkpoint
        resume_itr = self.pipeline.load()

        print('Setup Complete. Starting training...')

        for itr in range(resume_itr, self.iterations + 1):

            fetches = [train_op]

            feed_dict = {
                self.generator.handle: train_handle,
                outputs['training']: True
            }

            if itr % SUMMARY_INTERVAL == 0:
                fetches.append(total_loss)
                if self.summary_writer is not None:
                    fetches.append(train_summaries)

            start = time.time()
            result = sess.run(fetches, feed_dict)

            if itr % SUMMARY_INTERVAL == 0:
                print('Summary iter', itr, '| Loss:',
                      result[1], '| Time:', time.time() - start)
                if self.summary_writer is not None:
                    self.summary_writer.add_summary(sess, result[-1], itr)

            if (itr % VAL_SUMMARY_INTERVAL == 0 and
                        self.summary_writer is not None):
                feed_dict = {
                    self.generator.handle: validation_handle,
                    outputs['training']: False
                }
                result = sess.run([validation_summaries], feed_dict)
                self.summary_writer.add_summary(sess, result[0], itr)

            if itr % EVAL_INTERVAL == 0 and itr > 1 and self.eval is not None:
                acc = self.eval.evaluate(itr)
                print('Evaluation at iter %d. Success rate: %.2f' % (itr, acc))
                if self.summary_writer is not None:
                    eval_success = sess.run(
                        self.eval_summary, {self.eval_summary_in: acc})
                    self.summary_writer.add_summary(sess, eval_success, itr)

            if itr % SAVE_INTERVAL == 0:
                self.pipeline.save(itr)
