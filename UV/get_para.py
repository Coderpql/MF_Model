import pandas as pd

df_res = pd.DataFrame(columns=['K', 'Lamda'], index=range(25, 45, 5))
for p in range(25, 45, 5):
    path = 'res/k_fold_res/k_fold_mae_' + str(p) + '.csv'
    data = pd.read_csv(path)
    data = data.drop(columns=['Unnamed: 0'])
    min_mae = 1
    min_mae_index = 0
    for i in data.index:
        if data.loc[i, 'mae_k_fold'] < min_mae:
            min_mae = data.loc[i, 'mae_k_fold']
            min_mae_index = i
    df_res.loc[p, 'K'] = data.loc[min_mae_index, 'K']
    df_res.loc[p, 'Lamda'] = data.loc[min_mae_index, 'Lamda']
df_res.index.name = 'TR'
df_res.to_csv('res/uv_k_fold.csv')
