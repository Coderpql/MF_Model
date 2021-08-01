import pandas as pd
import numpy as np


# 加载数据
def load_data(path):
    data = pd.read_csv(path)
    data.set_index(data['user_id'], inplace=True)
    data = data.drop(columns=['user_id'])
    return data


# 转换数据格式
def mat_to_chart(path_data, path_res):
    # 加载数据
    data = load_data(path_data)
    # 创建转换数据表格
    df_chart = pd.DataFrame(columns=['user_id', 'items_id', 'score'])
    # 数据转换
    for index in data.index:
        for column in data.columns:
            if data.loc[index, column] != 0:
                # 创建数据条
                df_slice = pd.DataFrame([{'user_id': index, 'items_id': column, 'score': data.loc[index, column]}])
                # 数据拼接
                df_chart = df_chart.append(df_slice, sort=False)
    # 存储转换后的数据
    df_chart.to_csv(path_res, index=False)


if __name__ == '__main__':
    # 转换训练验证数据
    for p in range(25, 45, 5):
        for c in ['', '_test']:
            print('Train and valid, p: %d, c: %s' % (p, c))
            path_Data = 'data_mat/target/0_' + str(p) + c + '.csv'
            path_Res = 'data_no_mat/target/0_' + str(p) + c + '.csv'
            mat_to_chart(path_Data, path_Res)

    # 转换交叉验证数据
    for p in range(25, 45, 5):
        for c in range(2):
            print('K_Fold, p: %d, c: %s' % (p, c))
            path_Data = 'data_mat/target_k_fold/0_' + str(p) + '_' + str(c) + '.csv'
            path_Res = 'data_no_mat/target_k_fold/0_' + str(p) + '_' + str(c) + '.csv'
            mat_to_chart(path_Data, path_Res)
