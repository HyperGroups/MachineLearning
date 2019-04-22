import sys
# sys.path.append('..')
# sys.path.append('../../')
import numpy as np
import PIL
import os
from myPath import *



if __name__=='__main__':
    ctx=mx.cpu()
    from myPath import *
    file_sym=os.path.join(root_dir, 'model-wolfram/wolfram.lstm-symbol.json')
    file_nd=os.path.join(root_dir, 'model-wolfram/wolfram.lstm-0000.params')

    _sym= mx.symbol.load(file_sym)
    _nd= mx.nd.load(file_nd)

    _nd['4.State']=mx.nd.array([[0,0,0,0,0]])
    _nd['4.CellState']=mx.nd.array([[0,0,0,0,0]])

    input_data=np.array([[1,2]])
    print 'input_data', input_data
    array = mx.nd.array(input_data)

    _nd["Input"] = array
    _e = _sym.bind(ctx, _nd)

    _out = _e.forward()
    prob = _out[0].asnumpy()
    prob = np.squeeze(prob)
    print 'prob', prob
