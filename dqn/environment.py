import numpy as np
import tensorflow as tf
import os
import h5py
import cv2
from dqn.utils import *

class Env(object):
    def __init__(self, config):
        self.reward = 0
        self.terminal = True
        self.stop_sp = config.stop_sp
        self.reward_func = config.reward_func
        self.train = config.train
        self.count = 0  # count restoration sp
        self.psnr, self.psnr_pre, self.psnr_init = 0., 0., 0.
        self.comparison_metric, self.comparison_metric_pre = 0., 0.
        self.reward_cal = config.reward_cal
        
        if self.train:
            # training data
            self.train_dir = config.train_dir
            f = h5py.File(self.train_dir + 'train.h5', 'r')
            self.data = f['data'].value
            self.label = f['label'].value
            f.close()
            self.data_index = 0
            self.data_len = len(self.data)

            # validation data
            f = h5py.File(config.val_dir + 'validation.h5', 'r')
            self.data_test = f['data'].value
            self.label_test = f['label'].value
            f.close()
            self.data_all = self.data_test
            self.label_all = self.label_test
        else:
            if config.new_image:
                self.my_img_dir = config.dataset
                self.my_img_list = os.listdir(self.my_img_dir)
                self.my_img_list.sort()
                self.my_img_idx = 0

            else:
                # test data
                self.test_batch = config.test_batch
                self.test_in = os.path.join(config.dataset, config.dataset.split(os.sep)[-1] + '_in/')
                self.test_gt = os.path.join(config.dataset, config.dataset.split(os.sep)[-1] + '_gt/')
                #print(self.test_in)
                #print(self.test_gt)
                list_in = [self.test_in + name for name in os.listdir(self.test_in)]
                list_in.sort()
                list_gt = [self.test_gt + name for name in os.listdir(self.test_gt)]
                list_gt.sort()
                self.name_list = [os.path.splitext(os.path.basename(file))[0] for file in list_in]
                self.data_all, self.label_all = load_imgs(list_in, list_gt)
                self.test_total = len(list_in)
                self.test_cur = 0
    
                # data reformat, because the data for tools training are in a different format
                self.data_all = data_reformat(self.data_all)
                self.label_all = data_reformat(self.label_all)
                self.data_test = self.data_all[0 : min(self.test_batch, self.test_total), ...]
                self.label_test = self.label_all[0 : min(self.test_batch, self.test_total), ...]

        if self.train or not config.new_image:
            # input PSNR
            self.base_psnr = 0.
            for k in range(len(self.data_all)):
                self.base_psnr += calculate_psnr(self.data_all[k, ...], self.label_all[k, ...])
            self.base_psnr /= len(self.data_all)

            # reward functions
            self.rewards = {'sp_reward': sp_reward}
            self.reward_function = self.rewards[self.reward_func]

        # build toolbox
        self.action_size = 12 + 1
        toolbox_path = 'toolbox/'
        self.graphs = []
        self.sessions = []
        self.inputs = []
        self.outputs = []
        for idx in range(12):
            g = tf.Graph()
            with g.as_default():
                # load graph
                saver = tf.train.import_meta_graph(toolbox_path + 'tool%02d' % (idx + 1) + '.meta')
                # input data
                input_data = g.get_tensor_by_name('Placeholder:0')
                self.inputs.append(input_data)
                # get the output
                output_data = g.get_tensor_by_name('sum:0')
                self.outputs.append(output_data)
                # save graph
                self.graphs.append(g)
            sess = tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=True))
            with g.as_default():
                with sess.as_default():
                    saver.restore(sess, toolbox_path + 'tool%02d' % (idx + 1))
                    self.sessions.append(sess)


    def new_image(self):
        self.terminal = False
        while self.data_index < self.data_len:
            self.img = self.data[self.data_index: self.data_index + 1, ...]
            self.img_gt = self.label[self.data_index: self.data_index + 1, ...]
            self.psnr = calculate_psnr(self.img, self.img_gt)
            if self.psnr > 50:  # ignore too smooth samples and rule out 'inf'
                self.data_index += 1
            else:
                break

        # update training file
        if self.data_index >= self.data_len:
            # start from beginning
            self.data_index = 0
            while True:
                self.img = self.data[self.data_index: self.data_index + 1, ...]
                self.img_gt = self.label[self.data_index: self.data_index + 1, ...]
                self.psnr = calculate_psnr(self.img, self.img_gt)
                if self.psnr > 50:  # ignore too smooth samples and rule out 'inf'
                    self.data_index += 1
                else:
                    break

        self.reward = 0
        self.count = 0
        self.psnr_init = self.psnr
        self.data_index += 1
        return self.img, self.reward, 0, self.terminal


    def act(self, action):
        self.psnr_pre = self.psnr
        self.comparison_metric_pre = self.comparison_metric
        if action == self.action_size - 1:  # stop
            self.terminal = True
        else:
            feed_dict = {self.inputs[action]: self.img}
            with self.graphs[action].as_default():
                with self.sessions[action].as_default():
                    im_out = self.sessions[action].run(self.outputs[action], feed_dict=feed_dict)
            self.img = im_out

        if self.reward_cal == 'psnr':
            self.comparison_metric = calculate_psnr(self.img, self.img_gt)
        elif self.reward_cal == 'ssim':
            self.comparison_metric = calculate_ssim(self.img, self.img_gt)
        elif self.reward_cal == 'nrmse':
            self.comparison_metric = calculate_nrmse(self.img, self.img_gt)
        elif self.reward_cal == 'mse':
            self.comparison_metric = calculate_mse(self.img, self.img_gt)

        self.psnr = calculate_psnr(self.img, self.img_gt)

        # max sp
        if self.count >= self.stop_sp - 1:
            self.terminal = True

        # stop if too bad
        if self.psnr < self.psnr_init:
            self.terminal = True

        # calculate reward
        self.reward = self.reward_function(self.comparison_metric, self.comparison_metric_pre)
        self.count += 1

        return self.img, self.reward, self.terminal


    def act_test(self, action, sp = 0):
        reward_all = np.zeros(action.shape)
        psnr_all = np.zeros(action.shape)
        if sp == 0:
            self.test_imgs = self.data_test.copy()
            self.test_temp_imgs = self.data_test.copy()
            self.test_pre_imgs = self.data_test.copy()
            self.test_sps = np.zeros(len(action), dtype=int)
        for k in range(len(action)):
            img_in = self.data_test[k:k+1,...].copy() if sp == 0 else self.test_imgs[k:k+1,...].copy()
            img_label = self.label_test[k:k+1,...].copy()
            self.test_temp_imgs[k:k+1,...] = img_in.copy()

            if self.reward_cal == 'psnr':
                comparison_metric_pre = calculate_psnr(img_in, img_label)
            elif self.reward_cal == 'ssim':
                comparison_metric_pre = calculate_ssim(img_in, img_label)
            elif self.reward_cal == 'nrmse':
                comparison_metric_pre = calculate_nrmse(img_in, img_label)
            elif self.reward_cal == 'mse':
                comparison_metric_pre = calculate_mse(img_in, img_label)

            psnr_pre = calculate_psnr(img_in, img_label)
            if action[k] == self.action_size - 1 or self.test_sps[k] == self.stop_sp: # stop action or already stop
                img_out = img_in.copy()
                self.test_sps[k] = self.stop_sp # terminal flag
            else:
                feed_dict = {self.inputs[action[k]]: img_in}
                with self.graphs[action[k]].as_default():
                    with self.sessions[action[k]].as_default():
                        with tf.device('/gpu:0'):
                            img_out = self.sessions[action[k]].run(self.outputs[action[k]], feed_dict=feed_dict)
                self.test_sps[k] += 1
            self.test_pre_imgs[k:k+1,...] = self.test_temp_imgs[k:k+1,...].copy()
            self.test_imgs[k:k+1,...] = img_out.copy()  # keep intermediate results
            
            if self.reward_cal == 'psnr':
                comparison_metric = calculate_psnr(img_out, img_label)
            elif self.reward_cal == 'ssim':
                comparison_metric = calculate_ssim(img_out, img_label)
            elif self.reward_cal == 'nrmse':
                comparison_metric = calculate_nrmse(img_in, img_label)
            elif self.reward_cal == 'mse':
                comparison_metric = calculate_mse(img_in, img_label)

            psnr = calculate_psnr(img_out, img_label)
            reward = self.reward_function(comparison_metric, comparison_metric_pre)
            psnr_all[k] = psnr
            reward_all[k] = reward

        if self.train:
            return reward_all.mean(), psnr_all.mean(), self.base_psnr
        else:
            return reward_all, psnr_all, self.base_psnr


    def update_test_data(self):
        self.test_cur = self.test_cur + len(self.data_test)
        test_end = min(self.test_total, self.test_cur + self.test_batch)
        if self.test_cur >= test_end:
            return False #failed
        else:
            self.data_test = self.data_all[self.test_cur: test_end, ...]
            self.label_test = self.label_all[self.test_cur: test_end, ...]

            # update base psnr
            self.base_psnr = 0.
            for k in range(len(self.data_test)):
                self.base_psnr += calculate_psnr(self.data_test[k, ...], self.label_test[k, ...])
            self.base_psnr /= len(self.data_test)
            return True #successful


    def act_test_new(self, my_img_cur, action):
        if action == self.action_size - 1:
            return my_img_cur.copy()
        else:
            if my_img_cur.ndim == 4:
                feed_img_cur = my_img_cur
            else:
                feed_img_cur = my_img_cur.reshape((1,) + my_img_cur.shape)
            my_img_next = self.sessions[action].run(self.outputs[action], feed_dict={self.inputs[action]: feed_img_cur})
            return my_img_next[0, ...]


    def update_test_new(self):
        """
        :return: (image, image name) or (None, None)
        """
        if self.my_img_idx >= len(self.my_img_list):
            return None, None
        else:
            img_name = self.my_img_list[self.my_img_idx]
            base_name, _ = os.path.splitext(img_name)
            my_img = cv2.imread(self.my_img_dir + img_name)
            #print(my_img)
            my_img = my_img[:,:,::-1] / 255.
            self.my_img_idx += 1
            return my_img, base_name


    def get_test_imgs(self):
        return self.test_imgs.copy()


    def get_test_sps(self):
        return self.test_sps.copy()


    def get_data_test(self):
        return self.data_test.copy()


    def get_test_info(self):
        return self.test_cur, len(self.data_test) # current image number & batch size
