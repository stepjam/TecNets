import tensorflow as tf
import os

PREFIX = "itr"


class Saver(object):

    def __init__(self, savedir='.', savetitle=''):
        self.savedir = savedir
        self.savefile = os.path.join(savedir, savetitle)
        self.saver = None

    def save(self, sess, itr):
        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=10)
        self.saver.save(sess, self.savefile + "_" + PREFIX + str(itr))
        print('Saved model at iteration', itr)


class Loader(object):

    def __init__(self, savedir='.', savetitle='', checkpoint=-1):
        self.savedir = savedir
        self.checkpoint = checkpoint
        self.savefile = os.path.join(savedir, savetitle)
        self.saver = None

    def load(self, sess):
        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=10)
        model_file = tf.train.latest_checkpoint(self.savedir)
        if model_file:
            ind1 = model_file.rfind('itr')
            if self.checkpoint > 0:
                model_file = model_file[:ind1] + PREFIX + str(self.checkpoint)
                resume_itr = self.checkpoint
            else:
                resume_itr = int(model_file[ind1 + len(PREFIX):])
            print("Restoring model weights from " + model_file)
            self.saver.restore(sess, model_file)
            return resume_itr
        raise RuntimeError('Could not find model file in: %s' % self.savedir)
