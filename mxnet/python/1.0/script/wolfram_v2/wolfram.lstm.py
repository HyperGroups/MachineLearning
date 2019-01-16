import mxnet as mx
import numpy as np
model_prefix='/Users/hypergroups/Nutstore/ProjectsOnline/MyProjects/MXNetFinal/1.0/model-wolfram/wolfram.lstm'
# model_prefix='/home/hypergroups/Nutstore/ProjectsOnline/MyProjects/MXNetFinal/1.0/model-wolfram/wolfram.lstm'

ctx=mx.cpu()
sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 0)
mod = mx.mod.Module(symbol=sym, context=ctx,data_names=['Input'],label_names=None)
mod.bind(data_shapes=[('Input', (1,2))],for_training=False,
         label_shapes=mod._label_shapes)

arg_params['4.State']=mx.nd.array([[0,0,0,0,0]])
arg_params['4.CellState']=mx.nd.array([[0,0,0,0,0]])

mod.set_params(arg_params, aux_params, allow_missing=True)