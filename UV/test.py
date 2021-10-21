import numpy as np
import pandas as pd

data = pd.read_csv('data_no_mat/target/0_25.csv')
columns = ['user_id', 'items_id', 'score']
ratings = data.groupby(columns[0]).agg(['mean'])[[columns[2]]]
print(ratings)

B_U = dict(zip(ratings.index, ratings['score']['mean'].astype(np.float32)))
print(B_U)

mean = pd.DataFrame.mean(data['score'], axis=0)
print(mean)