import mxnet as mx
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import numpy as np
from myPath import *

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

ctx = mx.cpu()

# dir_model='/Users/hypergroups/Data/Wolfram/Model-MXNet-11.3'
# dir_model='/Users/hypergroups/Nutstore/ProjectsOnline/MyProjects/MXNetFinal/1.0/model-wolfram'
# dir_model='/home/hypergroups/Nutstore/ProjectsOnline/MyProjects/MXNetFinal/1.0/model-wolfram'
dir_model=os.path.join(root_dir, 'model-wolfram')
model_prefix=os.path.join(dir_model, 'object_count')

print 'model_prefix', model_prefix

sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 0)
mod = mx.mod.Module(symbol=sym, context=ctx,data_names=['Input'],label_names=None)
mod.bind(for_training=False, data_shapes=[('Input', (1,3,224,224))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
# mod.set_params(arg_params, aux_params, allow_missing=True)

def get_image(url, show=False):
    # download and show the image
    # fname = mx.test_utils.download(url)
    fname=url
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    if img is None:
         return None
    if show:
         plt.imshow(img)
         plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = 1 - np.asarray(img, dtype=np.uint8).astype(np.float32)/255
    print img.shape
    meanImage=[0,0,0]
    meanImage=map(lambda x:x, meanImage)

    img= map(lambda x:x - meanImage, img)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

def pred_file(url):
    print 'file@================', url
    img = get_image(url, show=True)
    # print img
    # compute the pred_file probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    print 'prob',prob
    print 'prob_sort', sorted(prob,reverse=True)
    idx = np.argsort(prob)[::-1]
    print 'top_idx',idx
    # np.savetxt('res.txt', prob)

if __name__=='__main__':

    file_pred=os.path.join(root_dir,'../Data/me_part.jpg')
    pred_file(file_pred)

    file_pred=os.path.join(root_dir,'../Data/me.jpg')
    pred_file(file_pred)

    file_pred=os.path.join(root_dir,'../Data/horse.jpg')
    pred_file(file_pred)

    file_pred=os.path.join(root_dir,'../Data/little_girl.jpg')
    pred_file(file_pred)