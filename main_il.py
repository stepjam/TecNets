from tensorflow.python.platform import flags
from data.mil_sim_reach import MilSimReach
from data.mil_sim_push import MilSimPush
from consumers.control import Control
from consumers.eval_consumer import EvalConsumer
from consumers.generator_consumer import GeneratorConsumer
from consumers.imitation_loss import ImitationLoss
from consumers.margin_loss import MarginLoss
from consumers.task_embedding import TaskEmbedding
from data.data_sequencer import DataSequencer
from data.generator import Generator
from networks.cnn import CNN
from trainers.il_trainer import ILTrainer
from trainers.pipeline import Pipeline
from trainers.summary_writer import SummaryWriter
import os
from networks.save_load import Saver, Loader

# Dataset/method options
flags.DEFINE_string(
    'dataset', 'sim_reach', 'One of sim_reach, sim_push.')

# Training Options
flags.DEFINE_integer(
    'iterations', 500000, 'The number of training iterations.')
flags.DEFINE_integer(
    'batch_size', 64, 'The number of tasks sampled per batch (aka batch size).')
flags.DEFINE_float(
    'lr', 0.0001, 'The learning rate.')
flags.DEFINE_integer(
    'support', 5, 'The number of support examples per task (aka k-shot).')
flags.DEFINE_integer(
    'query', 5, 'The number of query examples per task.')
flags.DEFINE_integer(
    'embedding', 20, 'The embedding size.')

# Model Options
flags.DEFINE_string(
    'activation', 'relu', 'One of relu, elu, or leaky_relu.')
flags.DEFINE_bool(
    'max_pool', False, 'Use max pool rather than strides.')
flags.DEFINE_list(
    'filters', [32, 64], 'List of filters per convolution layer.')
flags.DEFINE_list(
    'kernels', [3, 3], 'List of kernel sizes per convolution layer.')
flags.DEFINE_list(
    'strides', [2, 2], 'List of strides per convolution layer. '
                       'Can be None if using max pooling.')
flags.DEFINE_list(
    'fc_layers', [64, 64], 'List of fully connected nodes per layer.')
flags.DEFINE_float(
    'drop_rate', 0.0, 'Dropout probability. 0 for no dropout.')
flags.DEFINE_string(
    'norm', None, 'One of layer, batch, or None')

# Loss Options
flags.DEFINE_float(
    'lambda_embedding', 1.0, 'Lambda for the embedding loss.')
flags.DEFINE_float(
    'lambda_support', 1.0, 'Lambda for the support control loss.')
flags.DEFINE_float(
    'lambda_query', 1.0, 'Lambda for the query control loss.')
flags.DEFINE_float(
    'margin', 0.1, 'The margin for the embedding loss.')

# Logging, Saving, and Eval Options
flags.DEFINE_bool(
    'summaries', True, 'If false do not write summaries (for tensorboard).')
flags.DEFINE_bool(
    'save', True, 'If false do not save network weights.')
flags.DEFINE_bool(
    'load', False, 'If we should load a checkpoint.')
flags.DEFINE_string(
    'logdir', '/tmp/data', 'The directory to store summaries and checkpoints.')
flags.DEFINE_bool(
    'eval', False, 'If evaluation should be done.')
flags.DEFINE_integer(
    'checkpoint_iter', -1, 'The checkpoint iteration to restore '
                           '(-1 for latest model).')
flags.DEFINE_string(
    'checkpoint_dir', None, 'The checkpoint directory.')
flags.DEFINE_bool(
    'no_mujoco', True, 'Run without Mujoco. Eval should be False.')

FLAGS = flags.FLAGS

if not FLAGS.no_mujoco:
    from evaluation.eval_mil_reach import EvalMilReach
    from evaluation.eval_mil_push import EvalMilPush

filters = list(map(int, FLAGS.filters))
kernels = list(map(int, FLAGS.kernels))
strides = list(map(int, FLAGS.strides))
fc_layers = list(map(int, FLAGS.fc_layers))

data = None
if FLAGS.dataset == 'sim_reach':
    data = MilSimReach()
elif FLAGS.dataset == 'sim_push':
    data = MilSimPush()
else:
    raise RuntimeError('Unrecognised dataset.')

loader = saver = None
if FLAGS.save:
    saver = Saver(savedir=FLAGS.logdir)
if FLAGS.load:
    loader = Loader(savedir=FLAGS.logdir,
                    checkpoint=FLAGS.checkpoint_iter)

net = CNN(filters=filters,
          fc_layers=fc_layers,
          kernel_sizes=kernels,
          strides=strides,
          max_pool=FLAGS.max_pool,
          drop_rate=FLAGS.drop_rate,
          norm=FLAGS.norm,
          activation=FLAGS.activation)

sequencer = DataSequencer('first_last', data.time_horizon)
gen = Generator(dataset=data,
                batch_size=FLAGS.batch_size,
                support_size=FLAGS.support,
                query_size=FLAGS.query,
                data_sequencer=sequencer)

generator_consumer = GeneratorConsumer(gen, data, FLAGS.support, FLAGS.query)
task_emb = TaskEmbedding(network=net,
                         embedding_size=FLAGS.embedding,
                         include_state=False,
                         include_action=False)
ml = MarginLoss(margin=FLAGS.margin, loss_lambda=FLAGS.lambda_embedding)
ctr = Control(network=net,
              action_size=data.action_size,
              include_state=True)
il = ImitationLoss(support_lambda=FLAGS.lambda_support,
                   query_lambda=FLAGS.lambda_query)
consumers = [generator_consumer, task_emb, ml, ctr, il]
p = Pipeline(consumers,
             saver=saver,
             loader=loader,
             learning_rate=FLAGS.lr)
train_outs = p.get_outputs()

summary_w = None
log_dir = os.path.join(FLAGS.logdir, 'no_state_action')
if FLAGS.summaries:
    summary_w = SummaryWriter(log_dir)

eval = None
if FLAGS.eval:
    disk_images = FLAGS.dataset != 'sim_to_real_place'
    econs = EvalConsumer(data, sequencer, FLAGS.support, disk_images)
    task_emb = TaskEmbedding(network=net,
                             embedding_size=FLAGS.embedding,
                             include_state=False,
                             include_action=False)
    ml = MarginLoss(margin=FLAGS.margin, loss_lambda=FLAGS.lambda_embedding)
    ctr = Control(network=net,
                  action_size=data.action_size,
                  include_state=True)
    peval = Pipeline([econs, task_emb, ml, ctr])
    outs = peval.get_outputs()
    if FLAGS.dataset == 'sim_reach':
        eval = EvalMilReach(sess=p.get_session(),
                            dataset=data,
                            outputs=outs,
                            supports=FLAGS.support,
                            num_tasks=10,
                            num_trials=10,
                            log_dir=log_dir,
                            record_gifs=True,
                            render=False)
    elif FLAGS.dataset == 'sim_push':
        eval = EvalMilPush(sess=p.get_session(),
                           dataset=data,
                           outputs=outs,
                           supports=FLAGS.support,
                           num_tasks=10,
                           num_trials=6,
                           log_dir=log_dir,
                           record_gifs=True,
                           render=False)
trainer = ILTrainer(pipeline=p,
                    outputs=train_outs,
                    generator=gen,
                    iterations=FLAGS.iterations,
                    summary_writer=summary_w,
                    eval=eval)
trainer.train()
