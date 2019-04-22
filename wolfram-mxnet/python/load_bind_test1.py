
import os
path_root=''
CONTEXT = {'device_type': 'cpu', 'device_id': 0}
ctx="cpu"

import numpy as np
import mxnet as mx

def load_model_bind(sym_file, nd_file, path_root='',ctx='cpu', gpu_id=CONTEXT['device_id']):
    sym_ = mx.symbol.load(sym_file)
    nd_ = mx.nd.load(os.path.join(path_root, nd_file))
    keys = nd_.keys()
    if ctx=="gpu":
        for key in keys:
            nd_[key] = mx.nd.array(nd_[key], ctx=mx.gpu(gpu_id))
    return [sym_, nd_]



def predict113(sym_,nd_,dataIn):
    dataInput = np.array([dataIn])
    # print 'dataInput', dataInput

    if ctx == "cpu":
        dataInputMX = mx.nd.array(dataInput, ctx=mx.cpu(CONTEXT['device_id']))
    elif ctx == "gpu":
        dataInputMX = mx.nd.array(dataInput, ctx=mx.gpu(CONTEXT['device_id']))
    else:
        print 'context error============================='

    # print img_inputND
    # print 'context@',img_inputND.context

    nd_["Input"] = dataInputMX
    # print dataInputMX
    nd_['4.State'] = mx.nd.array([[0, 0, 0, 0, 0]])
    nd_['4.CellState'] = mx.nd.array([[0, 0, 0, 0, 0]])

    e_ = sym_.bind(mx.cpu(0), nd_)
    out_ = e_.forward()
    prob = out_[0].asnumpy()[0]
    print prob

def predict120(sym_, nd_, dataIn):
    dataInput = np.array([dataIn])
    # print 'dataInput', dataInput

    if ctx == "cpu":
        dataInputMX = mx.nd.array(dataInput, ctx=mx.cpu(CONTEXT['device_id']))
    elif ctx == "gpu":
        dataInputMX = mx.nd.array(dataInput, ctx=mx.gpu(CONTEXT['device_id']))
    else:
        print 'context error============================='

    # print img_inputND
    # print 'context@',img_inputND.context


    nd_["Input"] = dataInputMX
    # print dataInputMX
    # nd_['Nodes'] = mx.nd.array((np.zeros(0,180)))
    nd_['Nodes'] = mx.nd.array(np.loadtxt('model-wolfram/wolfram_lstm-12.0.nodes.txt'))
    # nd_['Nodes'] = mx.nd.array(np.ones(180))

    nd_['4.State'] = mx.nd.array([[0,0,0,0,0]])
    nd_['4.CellState'] = mx.nd.array([[0,0,0,0,0]])
    # [-0.48563024 - 0.36583638  1.5399672]
    e_ = sym_.bind(mx.cpu(0), nd_)
    out_ = e_.forward()
    prob = out_[0].asnumpy()[0]
    print prob

file_sym=os.path.join(path_root, "model-wolfram/wolfram_lstm-12.0-symbol.json")
file_nd = os.path.join(path_root, "model-wolfram/wolfram_lstm-12.0-0000.params")

sym_,nd_=load_model_bind(file_sym, file_nd)
predict120(sym_,nd_, [1,2])

file_sym=os.path.join(path_root, "model-wolfram/wolfram_lstm-11.3-symbol.json")
file_nd = os.path.join(path_root, "model-wolfram/wolfram_lstm-11.3-0000.params")

sym_,nd_=load_model_bind(file_sym, file_nd)

predict113(sym_,nd_, [1,2])
