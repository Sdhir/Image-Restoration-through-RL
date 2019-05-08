# Data Generation

import cv2
import numpy
import numpy as np
import os, re
import theano
from theano import function
from theano.tensor.nnet.neighbours import images2neibs
from skimage import io
import h5py

mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')

def load_histology_image(imagepath, is_mask=0, scale=1.0) :
    curr_image = cv2.imread(imagepath)
    if scale != 1:
            curr_image = cv2.resize(curr_image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) 
    if(is_mask):
        curr_image[curr_image > 0] = 1
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        curr_image = np.asarray(curr_image, dtype = 'float32')
    else:
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
        curr_image = curr_image.transpose(2, 0, 1)
    return curr_image
    
def create_neibs(curr_image, blk_shape=(5,5), blk_step=(1,1)) :
    if( len(curr_image.shape) == 3):
        [img_chan, img_row, img_col] = curr_image.shape
        images = np.expand_dims(curr_image, axis=0)
    else :
        img_chan = 1;
        [img_row, img_col] = curr_image.shape
        images = np.expand_dims(np.expand_dims(curr_image, axis=0), axis=0)

    blk_func = function([], images2neibs(images, blk_shape, blk_step, mode='ignore_borders'), mode=mode_with_gpu)

    blk = blk_func()

    blk_count = len(blk)
    
    # Reshape so that blk's [[RGB], [RGB], ...]
    blk = blk.reshape((img_chan, blk_count//img_chan, -1)).transpose((1, 0, 2))
    blk = blk[:,].reshape((blk_count//img_chan, img_chan, blk_shape[0], blk_shape[1]))
    blk = blk.astype(np.float32)

    return blk

def data_generator(IMAGE_DIR):

    SCALE = 1
    PATCH_SHAPE = (63, 63)
    STRIDE_LEN = (31,31)

    tr_names = [f for f in os.listdir(IMAGE_DIR) if re.match(r'.+\.(jpg|jpeg|png)', f)]
    print("Generating patches...")
    for i, tr_name in enumerate(tr_names):    
        print("Working on image no: {}/ {} - {}".format(i+1,len(tr_names),tr_name))

        curr_image = load_histology_image(os.path.join(IMAGE_DIR, tr_name), 0, SCALE)
        for k in range(3):
            if k>0:
                curr_image = cv2.resize(curr_image, (0,0), fx=2/(k+2), fy=2/(k+2), interpolation=cv2.INTER_CUBIC) 
            neibs = create_neibs(curr_image, PATCH_SHAPE, STRIDE_LEN)
            for nn in range(neibs.shape[0]):
                im = neibs[nn].transpose(1,2,0)
                im = im.astype(np.uint8)
                # Apply noise sequencially
                #[blur_sigma,noise_sigma,JPEG_Q]
                temp = np.array([5,5,100])*np.random.rand(1,3)
                #Blur
                blur_img = cv2.GaussianBlur(im,(7,7),sigmaX=temp[0,0])
                #Noise
                noisy_img = blur_img + np.random.normal(0, temp[0,1], blur_img.shape)
                noisy_img_clipped = np.clip(noisy_img, 0, 255)  # we might get out of bounds due to noise
                #JPEG compression
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(temp[0,2])]
                _, encimg = cv2.imencode('.jpg', noisy_img_clipped, encode_param)
                decimg = cv2.imdecode(encimg, 1)
                
                if i ==0 and nn == 0:
                    data = decimg
                    data = data[np.newaxis,...]
                    label = im
                    label = label[np.newaxis,...]
                else:
                    data = np.concatenate((data,decimg[np.newaxis,...]),axis=0)
                    label = np.concatenate((label,im[np.newaxis,...]),axis=0)
    #[C,H,W,N]
    data = data.tranpose(3,1,2,0)
    label = label.tranpose(3,1,2,0)

    hf = h5py.File('train.h5', 'w')
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)
    hf.close()

if __name__ == '__main__':
    data_generator(r"/mnt/dfs/ssbw5/Research/ECE/stanleyj/Sudhir/Epth_Seg/compare_models/fig")