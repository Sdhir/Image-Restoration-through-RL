import os
import pprint
import inspect
import tensorflow as tf

def class_vars(obj):
    return {k:v for k, v in inspect.getmembers(obj)
            if not k.startswith('__') and not callable(k)}

class BaseModel(object):
    def __init__(self, config):
        self._saver = None
        self.config = config

        try:
            self._attrs = config.__dict__['__flags']
        except:
            self._attrs = class_vars(config)
        if config.train:
            pp = pprint.PrettyPrinter().pprint
            pp(self._attrs)

        self.config = config

        for attr in self._attrs:
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(self.config, attr))

    def save_model_fn(self, sp=None):
        print("Saving model...")
        if not os.path.exists(self.save_model):
            os.makedirs(self.save_model)
        self.saver.save(self.sess, self.save_model, global_step=sp)

    def load_model_fn(self):
        print("Loading model...")
        ckpt = tf.train.get_checkpoint_state(self.load_model)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.load_model, ckpt_name)
            self.saver.restore(self.sess, fname)
            print("Load SUCCESS: {}".format(fname))
            return True
        else:
            print("Load FAILED!: {}".format(self.load_model))
            return False

    @property
    def saver(self):
        if self._saver == None:
            self._saver = tf.train.Saver(max_to_keep=2)
        return self._saver
