import pandas as pd
import numpy as np

if __name__ == '__main__':
    data_prov = pd.read_csv('../data/k_fold/provTriplet.csv')

    data_0 = data_prov.loc[data_prov['score'] == 0]
    data_1 = data_prov.loc[data_prov['score'] == 1]

    num_0 = data_0.shape[0]
    num_1 = data_1.shape[0]

    print(num_0)
    print(num_1)