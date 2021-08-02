import pandas as pd
import numpy as np
from random import sample
import math

np.random.seed(0)


# 加载数据
def load_data(path):
    data = pd.read_csv(path)
    data.set_index(data['user_id'], inplace=True)
    data = data.drop(columns=['user_id'])
    return data


# 创建K折交叉验证数据
def k_fold(k, domain_id, path):
    # 读取数据
    path_data = path + str(domain_id) + '.csv'
    data = load_data(path_data)

    # 创建分割数据的dataframe
    L_data = locals()
    for i in range(k):
        L_data['data' + str(i)] = pd.DataFrame(index=data.index, columns=data.columns, dtype=np.float32)
        L_data['data' + str(i)] = L_data['data' + str(i)].fillna(0)

    # 设置分割比例
    percent = 1 / float(k)

    # 数据分割
    i = 0
    for index in data.index:
        print('Division , domain:', domain_id, 'index_id:', i)
        list_items = []
        for column in data.columns:
            if data.at[index, column] > 0:
                list_items.append(column)
        if len(list_items) >= k:
            num_rating = math.floor(len(list_items) * percent)
            for i in range(k):
                choose_items = sample(list_items, num_rating)
                list_items = [x for x in list_items if x not in choose_items]
                for col in choose_items:
                    df_num = pd.DataFrame((data != 0).astype('int').sum(axis=0))
                    if df_num.loc[col, 0] >= k:
                        L_data['data' + str(i)].at[index, col] = data.at[index, col]
                        print('Division , domain:', domain_id, 'num_rating:', num_rating, 'col:', col)
                    else:
                        list_items.append(col)
                        L_data['data' + str(i)].at[index, col] = data.at[index, col]
                        print('Division , domain:', domain_id, 'num_rating:', 1, 'col:', col)
        else:
            for i in range(k):
                L_data['data' + str(i)].at[index, list_items[0]] = data.at[index, list_items[0]]
                print('Division , domain:', domain_id, 'num_rating:', 1, 'col:', list_items[0])
        i += 1

    # 数据补充
    for i in range(k):
        df_num = pd.DataFrame((L_data['data' + str(i)] != 0).astype('int').sum(axis=0))
        for index in df_num.index:
            print('Fill Data , item:', index)
            if df_num.at[index, 0] == 0:
                # 在原始数据中随机选取数据进行填充
                df = data.loc[(data[index] != 0)]
                n = df.values.shape[0]
                pos = np.random.randint(0, n)
                pos = df.index[pos]
                L_data['data' + str(i)].at[pos, index] = data.at[pos, index]
    # # 数据存储(UBV)
    # for i in range(k):
    #     L_data['data' + str(i)].to_csv('res/domain_' + str(domain_id) + '/data_' + str(i) + '.csv')
    # 数据存储(UBV)
    for i in range(k):
        L_data['data' + str(i)] = L_data['data' + str(i)].astype('int')
        L_data['data' + str(i)].to_csv('res/domain_' + str(domain_id) + '/data_' + str(i) + '.csv')


# 创建K折交叉验证数据
def k_fold_svm(k, per):
    # 读取数据
    path_data = 'data/data_lite/target/0_' + str(per) + '.csv'
    data = load_data(path_data)

    # 创建分割数据的dataframe
    L_data = locals()
    for i in range(k):
        L_data['data' + str(i)] = pd.DataFrame(index=data.index, columns=data.columns, dtype=np.int)
        L_data['data' + str(i)] = L_data['data' + str(i)].fillna(0)

    # 设置分割比例
    percent = 1 / float(k)

    # 数据分割
    ind = 0
    for index in data.index:
        print('Division , percent:', per, 'index_id:', i)
        list_items = []
        for column in data.columns:
            if data.at[index, column] > 0:
                list_items.append(column)
        if len(list_items) >= k:
            num_rating = math.floor(len(list_items) * percent)
            for i in range(k):
                choose_items = sample(list_items, num_rating)
                list_items = [x for x in list_items if x not in choose_items]
                for col in choose_items:
                    df_num = pd.DataFrame((data != 0).astype('int').sum(axis=0))
                    if df_num.loc[col, 0] >= k:
                        L_data['data' + str(i)].at[index, col] = data.at[index, col]
                        print('Division , percent:', per, 'num_rating:', num_rating, 'col:', col)
                    else:
                        list_items.append(col)
                        L_data['data' + str(i)].at[index, col] = data.at[index, col]
                        print('Division , percent:', per, 'num_rating:', 1, 'col:', col)
        else:
            for i in range(k):
                L_data['data' + str(i)].at[index, list_items[0]] = data.at[index, list_items[0]]
                print('Division , percent:', per, 'num_rating:', 1, 'col:', list_items[0])
        ind += 1

    # 数据补充
    for i in range(k):
        df_num = pd.DataFrame((L_data['data' + str(i)] != 0).astype('int').sum(axis=0))
        for index in df_num.index:
            print('Fill Data , item:', index)
            if df_num.at[index, 0] == 0:
                # 在原始数据中随机选取数据进行填充
                df = data.loc[(data[index] != 0)]
                n = df.values.shape[0]
                pos = np.random.randint(0, n)
                pos = df.index[pos]
                L_data['data' + str(i)].at[pos, index] = data.at[pos, index]
    # 数据存储(SVM)
    for i in range(k):
        L_data['data' + str(i)].to_csv('data/data_lite/target_k_fold/0_' + str(per) + '_' + str(i) + '.csv')


# 判断数据是否符合二折交叉验证要求
def Is_enough(path, k, domain_index):
    print('=====================================')
    for i in range(k):
        data = pd.read_csv(path + '/data_' + str(i) + '.csv')
        data.set_index(data['user_id'], inplace=True)
        data = data.drop(columns=['user_id'])
        n_rating = np.count_nonzero(data.values)
        num_u = (data != 0).astype('int').sum(axis=1)
        num_i = (data != 0).astype('int').sum(axis=0)
        print('domain:', domain_index, 'k:', i, 'n_rating:', n_rating)
        n = 0
        for u in num_u:
            if u == 0:
                n += 1
        if n == 0:
            print("u_ok")
        n = 0
        for i in num_i:
            if i == 0:
                n += 1
        if n == 0:
            print("i_ok")
    print('=====================================')


# 判断数据是否符合二折交叉验证要求
def Is_enough_svm(k, percent):
    print('=====================================')
    for i in range(k):
        data = pd.read_csv('data/data_lite/target_k_fold/0_' + str(percent) + '_' + str(i) + '.csv')
        data.set_index(data['user_id'], inplace=True)
        data = data.drop(columns=['user_id'])
        n_rating = np.count_nonzero(data.values)
        num_u = (data != 0).astype('int').sum(axis=1)
        num_i = (data != 0).astype('int').sum(axis=0)
        print('percent:', percent, 'k:', i, 'n_rating:', n_rating)
        n = 0
        for u in num_u:
            if u == 0:
                n += 1
        if n == 0:
            print("u_ok")
        n = 0
        for i in num_i:
            if i == 0:
                n += 1
        if n == 0:
            print("i_ok")
    print('=====================================')


if __name__ == '__main__':
    K = 2
    path_Data = 'data/data_lite/source/'
    # 分割UBV交叉验证数据
    # for d in range(1, 7):
    #     k_fold(K, d, path_Data)
    # print('finished!')

    # 查看分割数据是否合规
    # for d in range(1, 7):
    #     path_data = 'res/domain_' + str(d) + '/'
    #     Is_enough(path_data, K, d)

    # 分割svm交叉验证数据
    # for i in [25, 30, 35, 40]:
    #     k_fold_svm(2, i)
    # print('finished!')

    # 查看分割数据是否合规
    for i in [25, 30, 35, 40]:
        Is_enough_svm(2, i)