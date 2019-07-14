from unittest import main, TestCase
from evaluation.eval_mil_reach import EvalMilReach
from consumers.eval_consumer import EvalConsumer
from consumers.control import Control
from consumers.generator_consumer import GeneratorConsumer
from consumers.imitation_loss import ImitationLoss
from consumers.margin_loss import MarginLoss
from consumers.task_embedding import TaskEmbedding
from data.data_sequencer import DataSequencer
from data.generator import Generator
from data.mil_sim_push import MilSimPush
from data.mil_sim_reach import MilSimReach
from networks.cnn import CNN
from trainers.il_trainer import ILTrainer
from trainers.pipeline import Pipeline
import tensorflow as tf


class TestIntegration(TestCase):

    def _default_pipeline(self, dataset, q_s_size=2):
        support_size = query_size = q_s_size
        net = CNN(filters=[4, 4, 4, 4], fc_layers=[20, 20],
                  kernel_sizes=[3, 3, 3, 3], strides=[2, 2, 2, 2],
                  max_pool=False, norm=None, activation='relu')

        sequencer = DataSequencer('first_last', dataset.time_horizon)
        gen = Generator(dataset=dataset, batch_size=4,
                        support_size=support_size, query_size=query_size,
                        data_sequencer=sequencer)

        gen_con = GeneratorConsumer(gen, dataset, support_size, query_size)
        task_emb = TaskEmbedding(
            network=net, embedding_size=6, support_size=support_size,
            query_size=query_size, include_state=False, include_action=False)
        ml = MarginLoss(margin=0.1)
        ctr = Control(network=net, action_size=dataset.action_size,
                      include_state=True)
        il = ImitationLoss()
        consumers = [gen_con, task_emb, ml, ctr, il]
        p = Pipeline(consumers)
        outputs = p.get_outputs()
        trainer = ILTrainer(pipeline=p, outputs=outputs,
                            generator=gen, iterations=10)
        trainer.train()

    def _default_evaluation(self, dataset, q_s_size=2, disk_images=True):
        support_size = query_size = q_s_size
        net = CNN(filters=[4, 4, 4, 4], fc_layers=[20, 20],
                  kernel_sizes=[3, 3, 3, 3], strides=[2, 2, 2, 2],
                  max_pool=False, norm=None, activation='relu')
        sequencer = DataSequencer('first_last', dataset.time_horizon)
        eval_cons = EvalConsumer(dataset, sequencer, support_size, disk_images)
        task_emb = TaskEmbedding(
            network=net, embedding_size=6, support_size=support_size,
            query_size=query_size, include_state=False, include_action=False)
        ml = MarginLoss(margin=0.1)
        ctr = Control(network=net, action_size=dataset.action_size,
                      include_state=True)
        consumers = [eval_cons, task_emb, ml, ctr]
        p = Pipeline(consumers)
        outs = p.get_outputs()
        return outs, p.get_session()

    def test_imitation_learning_mil_reach(self):
        data = MilSimReach()
        with tf.variable_scope('test_imitation_learning_mil_reach'):
            self._default_pipeline(data, 1)

    def test_imitation_learning_mil_push(self):
        data = MilSimPush()
        with tf.variable_scope('test_imitation_learning_mil_push'):
            self._default_pipeline(data)

    def test_eval_mil_reach(self):
        data = MilSimReach()
        with tf.variable_scope('test_eval_mil_reach'):
            outs, sess = self._default_evaluation(data, 2)
            eval = EvalMilReach(sess=sess,
                                dataset=data,
                                outputs=outs,
                                supports=2,
                                num_tasks=2,
                                num_trials=2,
                                record_gifs=False,
                                render=False)
            tf.global_variables_initializer().run()
            eval.evaluate(0)


if __name__ == '__main__':
    main()
