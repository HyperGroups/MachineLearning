import os
import mxnet as mx

print __file__
root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print 'root_dir',root_dir
# print 'mxnet version=', mx.__version__