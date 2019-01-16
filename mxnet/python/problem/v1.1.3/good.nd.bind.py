import mxnet as mx
import os
import numpy as np


def test_lstm():
    print 'test@lstm'
    ctx = mx.cpu()
    file_sym = '/Users/hypergroups/Desktop/wolfram.lstm-symbol.json'
    file_nd = '/Users/hypergroups/Desktop/wolfram.lstm-0000.params'
    _sym = mx.symbol.load(file_sym)

    _nd = mx.nd.load(file_nd)

    input_data = np.array([[1, 2]])

    print input_data

    array = mx.nd.array(input_data)

    _nd["Input"] = array
    _nd['4.State'] = mx.nd.array([[0, 0, 0, 0, 0]])
    _nd['4.CellState'] = mx.nd.array([[0, 0, 0, 0, 0]])

    _e = _sym.bind(ctx, _nd)

    _out = _e.forward()
    prob = _out[0].asnumpy()
    prob = np.squeeze(prob)
    print 'prob', prob



if __name__ == '__main__':
    test_lstm()