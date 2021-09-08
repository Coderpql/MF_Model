import numpy as np
import pandas as pd
import yaml
from model.FUNK_SVD import FUNK_SVD
import pickle

if __name__ == '__main__':
    # load data
    data_train = pd.read_csv('../data/train/TR90Triplet.csv')
    data_test = pd.read_csv('../data/test/TE10Triplet.csv')

    # load config file
    path_config = '../config/config.yaml'
    with open(path_config, 'r', encoding='utf-8') as f:
        config = f.read()
    config = yaml.load(config, yaml.FullLoader)['FUNK_SVD']

    # create model
    model = FUNK_SVD(config, data_train, 2, 6)

    # train
    path_log = '../res/log/'
    model.fit()

    # test
    mae, rmse = model.test(data_test)
    print('Test  mae: {:.4f}, rmse: {:.4f}'.format(mae, rmse))
    # save
    model_file = open("../res/model/FUNK_SVD_model.pkl", "wb")
    pickle.dump(model, model_file)
    model_file.close()

    # load model
    model_file = open("../res/model/FUNK_SVD_model.pkl", "rb")
    model = pickle.load(model_file)
    mae, rmse = model.test(data_test)
    print('Test  mae: {:.4f}, rmse: {:.4f}'.format(mae, rmse))
