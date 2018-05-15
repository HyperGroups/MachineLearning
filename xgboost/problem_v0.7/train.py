import sys
import xgboost as xgb
import os
import numpy as np


file_train='train'
file_test='test'
file_model='model_lr'

param = {

        'bst:eta': 0.5, 'silent': 0, 'objective': 'binary:logistic',
        'booster': 'gblinear',
        'alpha': 5.00,
        'lambda': 10.00,
        'lambda_bias': .4,

}


###

dtrain = xgb.DMatrix(file_train)
dtest = xgb.DMatrix(file_test)


def train():
    print 'parameters', param
    print 'file_model', file_model
    plst = param.items()
    plst += [('eval_metric', 'rmse')]  # Multiple evals can be handled in this way
    plst += [('eval_metric', 'auc')]  # Multiple evals can be handled in this way
    plst += [('eval_metric', 'error')]  # Multiple evals can be handled in this way
    plst += [('eval_metric', 'logloss')] 
    evallist = [(dtrain, 'train'), (dtest, 'test')]

    num_round_lr=5
    model = xgb.train(plst, dtrain, num_round_lr, evallist)
    model.save_model( file_model )
    model.dump_model(file_model + '.dump.txt')

if __name__ == '__main__':
    train()
