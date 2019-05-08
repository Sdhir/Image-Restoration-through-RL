import random
import tensorflow as tf
from dqn.agent import Restore_agent
from dqn.environment import Env
import config
import sys

flags = tf.app.flags
flags.DEFINE_boolean('gpu', True, 'Use GPU')
flags.DEFINE_boolean('train', False, 'training or testing')
# training
flags.DEFINE_string('save_model', 'path/to/models/', 'Path to save models')
flags.DEFINE_string('logs', 'path/to/logs/', 'Path to tensorboard logs')
# test
flags.DEFINE_boolean('save_res', True, 'save results')
flags.DEFINE_string('dataset', 'path/to/data', 'Path to test data')
flags.DEFINE_boolean('new_image', False, 'Test image without groundtruth')
flags.DEFINE_string('load_model', 'path/to/models/', 'Path to load model')

ARGS = flags.FLAGS

def main(_):
    with tf.Session() as sess:
        configure = config.get_config(ARGS)
        env = Env(configure)
        agent = Restore_agent(configure, env, sess)
        #sys.exit(0)

        if ARGS.train:
            print("Training...")
            agent.train_model()
        else:
            print("Testing...")
            if ARGS.new_image:
                agent.restore_newIm()
            else:
                agent.restore()


if __name__ == '__main__':
    tf.app.run()
