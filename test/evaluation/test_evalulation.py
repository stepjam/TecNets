from unittest import main, TestCase
import tensorflow as tf
from evaluation.eval_mil_reach import EvalMilReach
from evaluation.eval_mil_push import EvalMilPush
from consumers import eval_consumer
from data import data_sequencer
from data.mil_sim_reach import MilSimReach
from data.mil_sim_push import MilSimPush


class TestEvaluation(TestCase):

    def test_sim_reach(self):

        data = MilSimReach()
        with tf.variable_scope('test_sim_reach'):
            data_seq = data_sequencer.DataSequencer('first_last',
                                                    data.time_horizon)
            eval_con = eval_consumer.EvalConsumer(data, data_seq, 2)
            outputs = eval_con.consume({})
            outputs['sentences'] = tf.ones((1, 20))
            outputs['output_actions'] = tf.ones((1, data.action_size))
            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            eval = EvalMilReach(sess, dataset=data, outputs=outputs, supports=2,
                                num_tasks=2, num_trials=2, log_dir=".",
                                record_gifs=False, render=False)
            eval.evaluate(0)

    def test_sim_push(self):

        data = MilSimPush()
        with tf.variable_scope('test_sim_push'):
            data_seq = data_sequencer.DataSequencer('first_last',
                                                    data.time_horizon)
            eval_con = eval_consumer.EvalConsumer(data, data_seq, 2)
            outputs = eval_con.consume({})
            outputs['sentences'] = tf.ones((1, 20))
            outputs['output_actions'] = tf.ones((1, data.action_size))
            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            eval = EvalMilPush(sess, dataset=data, outputs=outputs, supports=2,
                                num_tasks=2, num_trials=2, log_dir=".",
                                record_gifs=False, render=False)
            eval.evaluate(0)


if __name__ == '__main__':
    main()
