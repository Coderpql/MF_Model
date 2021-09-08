import sys
sys.path.append('../model/')
import numpy as np
import pandas as pd
import yaml
from model.FUNK_SVD import FUNK_SVD
import pickle

if __name__ == '__main__':
    # load data
    # create the table of result
    col_names = ['TR', 'mae', 'rmse']
    result = pd.DataFrame(columns=col_names)
    result.to_csv('../res/log/res.csv', index=False)
    for TR in [60, 70, 80, 90]:
        TE = 100 - TR
        data_train = pd.read_csv('../data/train/TR' + str(TR) + 'Triplet.csv')
        data_test = pd.read_csv('../data/test/TE' + str(TE) + 'Triplet.csv')

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
        model_file = open("../res/model/FUNK_SVD_model_TR" + str(TR) + ".pkl", "wb")
        pickle.dump(model, model_file)
        model_file.close()

        # save the result
        result_slice = pd.DataFrame(
            {'TR': [TR], 'mae': [mae], 'rmse': [rmse]})
        result_slice.to_csv('../res/log/res.csv', index=False, mode='a', header=False)
