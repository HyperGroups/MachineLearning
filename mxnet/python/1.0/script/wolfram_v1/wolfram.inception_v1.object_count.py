import sys
sys.path.append('..')
# sys.path.append('../../')
import numpy as np
import PIL
import os
from myPath import *
# from myPath import root_dir
# print root_dir
# import myPath
# print '123',myPath.root_dir


# root_dir='/Users/hypergroups/Nutstore/ProjectsOnline/MyProjects/MXNetFinal/'
# root_dir='/home/hypergroups/Nutstore/ProjectsOnline/MyProjects/MXNetFinal/'
# root_dir='/home/sysadmin/nutshare/ProjectsOnline/MyProjects/MXNetFinal'


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(1, 3, 224, 224)


def pred_file(img_url):
    print 'file@================', img_url
    # manual 'NetEncoder' step
    img3 = PIL.Image.open(img_url)
    img = img3.resize((224, 224), PIL.Image.ANTIALIAS)
    img = 1 - np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).astype(np.float32)/255
    meanImage=[0,0,0]

    meanImage=map(lambda x: x, meanImage)

    img[0]=img[0]-meanImage[2]
    img[1]=img[1]-meanImage[1]
    img[2]=img[2]-meanImage[0]
    mxnet_img = to4d(img)

    img_inputND = mx.nd.array(mxnet_img)
    # print img_inputND

    _nd["Input"]= img_inputND
    _e = _sym.bind(mx.cpu(), _nd)
    _out = _e.forward()
    prob=_out[0].asnumpy()
    prob = np.squeeze(prob)
    print 'prob', prob
    print 'prob_sort', sorted(prob,reverse=True)
    idx = np.argsort(prob)[::-1]
    print 'top_idx',idx
    return _out

if __name__=='__main__':
    from myPath import *
    file_sym=os.path.join(root_dir, 'model-wolfram/object_count-symbol.json')
    file_nd=os.path.join(root_dir, 'model-wolfram/object_count-0000.params')

    _sym= mx.symbol.load(file_sym)

    _nd= mx.nd.load(file_nd)

    file_pred=os.path.join(root_dir,'../Data/me_part.jpg')
    pred_file(file_pred)

    file_pred=os.path.join(root_dir,'../Data/me.jpg')
    pred_file(file_pred)

    file_pred=os.path.join(root_dir,'../Data/horse.jpg')
    pred_file(file_pred)

    file_pred=os.path.join(root_dir,'../Data/little_girl.jpg')
    pred_file(file_pred)