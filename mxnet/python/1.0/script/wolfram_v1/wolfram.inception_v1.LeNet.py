import mxnet as mx
import numpy as np
import PIL
import os

root_dir='/Users/hypergroups/Nutstore/ProjectsOnline/My/Python/MXNet/'

file_sym=os.path.join(root_dir, '1.0/model-wolfram/object_count-symbol.json')
file_nd=os.path.join(root_dir, '1.0/model-wolfram/object_count-0000.params')


_sym= mx.symbol.load(file_sym)
_nd= mx.nd.load(file_nd)


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(1, 3, 224, 224)


def pred_file(img_url):
    # manual 'NetEncoder' step
    img3 = PIL.Image.open(img_url)
    img = img3.resize((224, 224), PIL.Image.ANTIALIAS)
    img = 1 - np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).astype(np.float32)/255
    mxnet_img = to4d(img)

    img_inputND = mx.nd.array(mxnet_img)
    # print img_inputND

    _nd["Input"]= img_inputND
    _e = _sym.bind(mx.cpu(), _nd)
    _out = _e.forward()
    # return np.argmax(out_object_count)

    print 'out', _out[0].asnumpy()
    return _out

# pred_object_count('/home/hypergroups/Projects/PycharmProjects/Data/horse.jpg')
pred_file(os.path.join(root_dir,'Data/8.jpg'))