import mxnet as mx
import numpy as np
model_prefix='/Users/hypergroups/Nutstore/ProjectsOnline/MyProjects/MXNetFinal/1.0/model-wolfram/example'

sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 0)
print sym, arg_params, aux_params

mod = mx.mod.Module(sym, context=mx.cpu(),data_names=['Input'],label_names=None)
label_shape=None
mod.bind(for_training=False,data_shapes=[('Input', (1, 2))], label_shapes=label_shape)
mod.set_params(arg_params, aux_params, allow_missing=True)

input_data=np.array([[1,2]])
print input_data
array = mx.nd.array(input_data)

mod.forward(array)

prob = mod.get_outputs()[0].asnumpy()
prob = np.squeeze(prob)
print prob