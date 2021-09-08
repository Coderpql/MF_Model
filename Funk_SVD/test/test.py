import numpy as np
import pandas as pd
import yaml
from model.FUNK_SVD import FUNK_SVD
import pickle

if __name__ == '__main__':
    # load data
    data_test = pd.read_csv('../data/test/TE10Triplet.csv')

    # load model
    model_file = open("../res/model/FUNK_SVD_model.pkl", "rb")
    model = pickle.load(model_file)
    mae, rmse = model.test(data_test)
    print('Test  mae: {:.4f}, rmse: {:.4f}'.format(mae, rmse))