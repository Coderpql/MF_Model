import pickle
import pandas as pd

if __name__ == '__main__':
    # load data
    columns = ['user_id', 'item_id', 'score']
    data_test = pd.read_csv('../data/target/test/TE10Triplet.csv', names=columns)

    # load model
    model_file = open("../res/model/TMF_model.pkl", "rb")
    model = pickle.load(model_file)
    mae, rmse = model.test(data_test)
    print('Test  mae: {:.4f}, rmse: {:.4f}'.format(mae, rmse))
