import sys
sys.path.append('..')
import mxnet as mx
import numpy as np
import PIL
from myPath import *
import os

dir_model='/mnt/hgfs/Share/Models/MXNet-Mma11.3/'
# dir_model='/mnt/hgfs/Share/Models/MXNet/'
# dir_model='/home/hypergroups/nutstore/Wolfram Mathematica/Data/NetModel@MXNet@V11.3/'

model_prefix=os.path.join(dir_model, 'Wolfram ImageIdentify Net V1')

print model_prefix
sym,arg_params, aux_params = mx.model.load_checkpoint(model_prefix,0)
print 'len(arg_params)', len(arg_params)
print 'sym.list_auxiliary_states@',len(sym.list_auxiliary_states()), \
    'sym.list_auxiliary_states@',sym.list_auxiliary_states()

print type(sym),sym.list_arguments()
print 'sym.list_auxiliary_states=',sym.list_auxiliary_states


def to4d(img, size=[224,224]):
    """
    reshape to 4D arraysr
    """
    return img.reshape(1, 3, size[0], size[1])


def pred_img(imgRGB, size=[224,224]):
    # manual 'NetEncoder' step

    img = 1 - np.asarray(imgRGB, dtype=np.uint8).transpose(2, 0, 1).astype(np.float32)/255
    mxnet_img = to4d(img, size=size)

    img_inputND = mx.nd.array(mxnet_img)
    # print img_inputND
    # arg_params={}
    for k in sym.list_auxiliary_states():
        if k not in aux_params:
            print k
            # aux_params[k]=mx.nd.array([[]])
    # for k in sym.list_arguments():
    #     arg_params[k]=mx.nd.array([[]])

    print 'len@arg_params=', len(aux_params)
    # arg_params['conv_batchnorm.MovingMean']=[[0]]
    arg_params["Input"]= img_inputND
    e = sym.bind(mx.cpu(), arg_params, aux_states=aux_params)
    out = e.forward()
    return np.argmax(out)


def pred_file(img_url, size=[224,224]):
    imgRGB = PIL.Image.open(img_url)
    imgResize = imgRGB.resize((size[0], size[1]), PIL.Image.ANTIALIAS)

    res=pred_img(imgResize, size=size)

    return res


if __name__ == "__main__":

    file_pred=os.path.join(root_dir,'../Data/horse.jpg')
    pred_file(file_pred)

    file_pred=os.path.join(root_dir,'../Data/little_girl.jpg')
    pred_file(file_pred)
