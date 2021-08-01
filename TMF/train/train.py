import pickle
import pandas as pd
import yaml
from model.TMF import TMF

if __name__ == '__main__':
    # load data
    columns = ['user_id', 'item_id', 'score']
    data_target = pd.read_csv('../data/target/train/TR90Triplet.csv', names=columns)
    data_source = pd.read_csv('../data/source/R2Triplet.csv', names=columns)
    data_test = pd.read_csv('../data/target/test/TE10Triplet.csv', names=columns)

    # load config file
    path_config = '../config/config.yaml'
    with open(path_config, 'r', encoding='utf-8') as f:
        config = f.read()
    config = yaml.load(config, yaml.FullLoader)['TMF']

    # create model
    model = TMF(config, data_target, data_source)
    # train
    path_log = '../res/log/'
    model.fit(path_log, data_test)
    # test
    mae, rmse = model.test(data_test)
    print('Test  mae: {:.4f}, rmse: {:.4f}'.format(mae, rmse))
    # save
    model_file = open("../res/model/TMF_model.pkl", "wb")
    pickle.dump(model, model_file)
    model_file.close()

    # load model
    model_file = open("../res/model/TMF_model.pkl", "rb")
    model = pickle.load(model_file)
    mae, rmse = model.test(data_test)
    print('Test  mae: {:.4f}, rmse: {:.4f}'.format(mae, rmse))
