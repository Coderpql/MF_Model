import math
import os
import numpy as np
import pandas as pd
from time import *
from itertools import combinations

from sklearn.impute import SimpleImputer

np.random.seed(0)


# data = np.array([[1, 1, 3, 1, 1],
#                  [3, 3, 2, 2, 3],
#                  [2, 2, 3, 3, 2],
#                  [1, 1, 3, 3, 1],
#                  [1, 1, 0, 3, 1],
#                  [3, 3, 2, 0, 3],
#                  [2, 2, 3, 3, 2]])

# 判断数据是否符合二折交叉验证要求
def Is_enough(domain_id, path):
    df_domain = pd.read_csv(path + str(domain_id) + '.csv')
    df_domain.set_index(df_domain['user_id'], inplace=True)
    df_domain = df_domain.drop(columns=['user_id'])

    num_u = (df_domain != 0).astype('int').sum(axis=1)
    num_i = (df_domain != 0).astype('int').sum(axis=0)
    print('=====================================')
    n = 0
    for u in num_u:
        if u < 2:
            n += 1
    print(n)
    n = 0
    for i in num_i:
        if i < 2:
            n += 1
    print(n)
    print('=====================================')


# 填充矩阵(最常出现的值)
def fill_matrix(M):
    imp = SimpleImputer(missing_values=0, strategy='most_frequent')  # mean  median  most_frequent
    M = imp.fit_transform(M)
    return M


# UBV分解训练
# @jit(nopython=True)
def train(M, U, B, V, max_iteration, source_id, k):
    print('UBV')
    Loss_old = float('inf')
    Loss_c = Loss_old
    iteration = 1
    L_loss = []
    while 1:
        # 计算梯度
        grad_U = np.sqrt(M.dot(V).dot(B.T) / U.dot(U.T).dot(M).dot(V).dot(B.T))
        grad_V = np.sqrt(M.T.dot(U).dot(B) / V.dot(V.T).dot(M.T).dot(U).dot(B))
        grad_B = np.sqrt(U.T.dot(M).dot(V) / U.T.dot(U).dot(B).dot(V.T).dot(V))

        # 更新参数
        U = U * grad_U
        V = V * grad_V
        B = B * grad_B

        # 计算损失值和损失值变化
        # Loss = np.square(np.linalg.norm((M - U.dot(B).dot(V.T))))
        Loss = np.linalg.norm((M - U.dot(B).dot(V.T)))
        L_loss.append(Loss)
        # Loss = np.sum(np.abs((M - U.dot(B).dot(V.T)))) / (M.shape[0] * M.shape[1])
        # Loss_c = Loss_old - Loss
        # Loss_old = Loss

        # 输出迭代信息
        print('Source_id', source_id, 'K:', k, 'Epoch:', iteration, 'Loss: ', Loss, 'Loss_Change: ', Loss_c)

        iteration += 1
        if iteration > max_iteration:
            break

    List_loss = np.array(L_loss)
    np.save('res/loss/loss_' + str(source_id) + '.npy', List_loss)
    return U, B, V


# 存储UBV分解后用户和商品的特征
def save_result(U, V, source_id):
    # 分解结果使用Dataframe进行存储
    df_U = pd.DataFrame(U)
    df_V = pd.DataFrame(V)
    # 将结果存储为csv格式数据
    df_U.to_csv('res/feature/feature_U_' + str(source_id) + '.csv', index=False)
    df_V.to_csv('res/feature/feature_V_' + str(source_id) + '.csv', index=False)


# 矩阵分解辅助域数据，并对用户和商品特征进行存储
def source_UBV(source_id, data, k_1, k_2, k):
    # 数据填充
    data = fill_matrix(data)
    # 显示填充后的数据
    # print(data)
    # 获取用户数和物品数
    m = data.shape[0]
    n = data.shape[1]
    # 随机初始化U、B、V矩阵
    M_U = np.random.rand(m, k_1).astype('float32')
    M_V = np.random.rand(n, k_2).astype('float32')
    M_B = np.random.rand(k_1, k_2).astype('float32')
    data = data.astype('float32')
    Epoch = 1000
    start = time()
    U_, B_, V_ = train(data, M_U, M_B, M_V, Epoch, source_id, k)
    end = time()
    M_predict = U_.dot(B_).dot(V_.T)
    # print('==================================U==================================')
    # print(U_)
    # print('==================================B==================================')
    # print(B_)
    # print('==================================V==================================')
    # print(V_)
    # print('===============================M_predict=============================')
    # print(M_predict)

    print('运行时间: ', end - start)

    # 存储用户和商品特征
    # save_result(U_, V_, source_id)

    return M_predict


# 生成辅助域组合
def create_source_combination(path):
    # 获取文件列表
    files = os.listdir(path_Data)
    files = list(range(1, len(files) + 1))
    # 对文件列表进行随机组合
    df_combination = pd.DataFrame(columns=range(2))
    for i in range(0, int(len(files) / 2)):
        for j in range(int(len(files) / 2), len(files)):
            data_slice = pd.DataFrame([{0: files[i], 1: files[j]}])
            df_combination = df_combination.append(data_slice)
    df_combination.index = (list(range(len(df_combination[0]))))
    print(df_combination)
    df_combination.to_csv('res/combination.csv', index=False)


# K折交叉验证
def verify_k_fold(k, path):
    path_train_data = ''
    path_test_data = ''
    k1 = [5, 10, 15, 20, 25, 30, 35, 40]
    k2 = [5, 10, 15, 20, 25, 30, 35, 40]
    for i in range(1, 7):
        for k_1 in k1:
            for k_2 in k2:
                mae = 0.0
                li = ['Domain', 'K1', 'K2', 'mae_k_fold']
                res = pd.DataFrame(columns=li)
                for k_i in range(k):
                    for k_j in range(k):
                        if k_i == k_j:
                            path_train_data = path + 'domain_' + str(i) + '/data_' + str(k_j) + '.csv'
                        else:
                            path_test_data = path + 'domain_' + str(i) + '/data_' + str(k_j) + '.csv'
                    # 加载训练数据
                    data_train = pd.read_csv(path_train_data)
                    data_train.set_index(data_train['user_id'], inplace=True)
                    data_train = data_train.drop(columns=['user_id']).values
                    # 加载测试数据
                    data_test = pd.read_csv(path_test_data)
                    data_test.set_index(data_test['user_id'], inplace=True)
                    data_test = data_test.drop(columns=['user_id']).values
                    # 创建测试集评分标记矩阵
                    mask = np.zeros((data_test.shape[0], data_test.shape[1]))
                    n_rating = 0
                    for mask_i in range(data_test.shape[0]):
                        for mask_j in range(data_test.shape[1]):
                            if data_test[mask_i, mask_j] != 0:
                                n_rating += 1
                                mask[mask_i, mask_j] = 1
                    # 对训练集进行矩阵分解训练
                    M_pred = source_UBV(i, data_train, k_1, k_2, k_i)
                    # 根据测试集评分标记矩阵对测试集对应数据进行保留
                    for mask_i in range(data_test.shape[0]):
                        for mask_j in range(data_test.shape[1]):
                            if mask[mask_i, mask_j] == 0:
                                M_pred[mask_i, mask_j] = 0
                    # 计算mae
                    mae += np.sum(np.abs(M_pred - data_test)) / n_rating

                mae /= k
                print('Domain:', i, 'k1', k_1, 'k2', k_2, 'mae:', mae)
                result = pd.DataFrame({'Domain': [i], 'K1': [k_1], 'K2': [k_2], 'mae_k_fold': [mae]})
                res = res.append(result, sort=False, ignore_index=True)
                res.to_csv('res/domain_' + str(i) + '/mae_k_fold.csv', mode='a', index=False, header=False)


# 构建分类数据
def create_class_data(M, path_data, comb, combination_id, percent, c):
    # 数据格式 [Lu, Lv, Fu, Fv, Score]
    f_u = None
    f_v = None
    l_u = 0
    l_v = 0
    score = 0
    line = None
    U = pd.read_csv(path_data + 'feature/feature_U_' + str(comb[0]) + '.csv').values
    V = pd.read_csv(path_data + 'feature/feature_V_' + str(comb[1]) + '.csv').values
    print(U)
    print(V)
    # 创建分类数据表格
    data_class = np.array(range(M.shape[0] + M.shape[1] + U.shape[1] + V.shape[1] + 1))
    # data_class = np.array(range(U.shape[1] + V.shape[1] + 1))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] != 0:
                # 生成one_hot形式用户商品编号
                l_u = np.zeros((1, U.shape[0]))
                l_u[0, i] = 1
                l_v = np.zeros((1, V.shape[0]))
                l_v[0, j] = 1
                # 提取用户商品特征
                f_u = U[i].reshape(1, -1)
                f_v = V[j].reshape(1, -1)
                # 提取分数
                score = np.array([M[i, j]])
                # 将提取的特征拼接为一条数据
                # line = np.concatenate((l_u, l_v, f_u, f_v, score), axis=0)
                # line = np.append(np.append(np.append(np.append(l_u, l_v), f_u), f_v), score)
                line = np.append(np.append(f_u, f_v), score)
                # 将获得的一条数据添加到数据表格中
                print('combination_id:', combination_id, 'percent:', percent, 'C:', c, 'user_id:', i, 'item_id:', j)
                data_class = np.vstack([data_class, line])
    # 数据存储
    data_class = data_class[1:, :]
    df_data_class = pd.DataFrame(data_class)
    # df_data_class.to_csv('res/class_data_no_pos/TR' + str(percent) + '/data_class_'
    #                      + str(combination_id) + c + '.csv', index=False)
    df_data_class.to_csv('res/class_data/TR' + str(percent) + '/data_class_'
                         + str(combination_id) + c + '.csv', index=False)
    print(df_data_class)


# 构建分类数据
def create_class_data_k_fold(M, path_data, comb, combination_id, percent, c):
    # 数据格式 [Lu, Lv, Fu, Fv, Score]
    f_u = None
    f_v = None
    l_u = 0
    l_v = 0
    score = 0
    line = None
    U = pd.read_csv(path_data + 'feature/feature_U_' + str(comb[0]) + '.csv').values
    V = pd.read_csv(path_data + 'feature/feature_V_' + str(comb[1]) + '.csv').values
    print(U)
    print(V)
    # 创建分类数据表格
    data_class = np.array(range(M.shape[0] + M.shape[1] + U.shape[1] + V.shape[1] + 1))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] != 0:
                # 生成one_hot形式用户商品编号
                l_u = np.zeros((1, U.shape[0]))
                l_u[0, i] = 1
                l_v = np.zeros((1, V.shape[0]))
                l_v[0, j] = 1
                # 提取用户商品特征
                f_u = U[i].reshape(1, -1)
                f_v = V[j].reshape(1, -1)
                # 提取分数
                score = np.array([M[i, j]])
                # 将提取的特征拼接为一条数据
                # line = np.concatenate((l_u, l_v, f_u, f_v, score), axis=0)
                line = np.append(np.append(np.append(np.append(l_u, l_v), f_u), f_v), score)
                # 将获得的一条数据添加到数据表格中
                print('combination_id:', combination_id, 'percent:', percent, 'C:', c, 'user_id:', i, 'item_id:', j)
                data_class = np.vstack([data_class, line])
    # 数据存储
    data_class = data_class[1:, :]
    df_data_class = pd.DataFrame(data_class)
    df_data_class.to_csv('res/class_data_k_fold/TR' + str(percent)
                         + '/data_class_' + str(combination_id) + '_' + str(c) + '.csv', index=False)
    print(df_data_class)


if __name__ == '__main__':
    path_Data = 'data/data_lite/source/'
    path_Res = 'res/'
    k = 2
    # verify_k_fold(k, path_Res)

    # for index in range(1, 7):
    #     Is_enough(index, path_Data)

    # 对各个辅助域进行矩阵分解，并对相应的分解特征进行存储
    for index in range(1, 7):
        data_train = pd.read_csv(path_Data + str(index) + '.csv')
        data_train.set_index(data_train['user_id'], inplace=True)
        data_train = data_train.drop(columns=['user_id']).values
        if index == 1:
            k1 = 40
            k2 = 40
            source_UBV(index, data_train, k1, k2, -1)
        else:
            k1 = 5
            k2 = 5
            source_UBV(index, data_train, k1, k2, -1)

    # 创建辅助域组合
    # create_source_combination(path_Data)

    # L_combination = pd.read_csv('res/combination.csv').values
    # print(L_combination)
    #
    # # 生成分类数据
    # index = 0
    # for combination in L_combination:
    #     for i in range(25, 45, 5):
    #         for j in ['', '_test']:
    #             data_0 = pd.read_csv('data/data_lite/target/0_' + str(i) + j + '.csv')
    #             data_0.set_index(data_0['user_id'], inplace=True)
    #             data_0 = data_0.drop(columns=['user_id']).values
    #             print(combination)
    #             create_class_data(data_0, path_Res, combination, index, i, j)
    #     index += 1

    # 生成分类数据(SVM_k_fold)
    # index = 0
    # for combination in L_combination:
    #     for i in range(25, 45, 5):
    #         for j in range(2):
    #             data_0 = pd.read_csv('data/data_lite/target_k_fold/0_' + str(i) + '_' + str(j) + '.csv')
    #             data_0.set_index(data_0['user_id'], inplace=True)
    #             data_0 = data_0.drop(columns=['user_id']).values
    #             print(combination)
    #             create_class_data_k_fold(data_0, path_Res, combination, index, i, j)
    #     index += 1
