import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer


def fill_data(file_name):  # 用来填充原始数据矩阵
    data = pd.read_csv(file_name)
    col_list = data.columns.tolist()  # 获取列名
    data = data.loc[:, col_list]  # 切片，去除数字index
    data = data.set_index('user_id')  # user_id设为index
    ind_list = data.index.tolist()  # 获取index，方便后面拼接
    col_list = data.columns.tolist()  # 获取columns
    data = data.loc[:, col_list]  # 重新切片，去除user_id列，方便填充，因为填充需要保留columns，去除index
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)  # mean  median  most_frequent
    data1 = pd.DataFrame(imp.fit_transform(data))  # 填充完成
    # imp = SimpleImputer(missing_values='NaN', strategy='most_frequent', axis=1)  # mean  median  most_frequent
    # data1 = pd.DataFrame(imp.fit_transform(data))  # 填充完成
    # 由于index和columns是数字，下面开始拼接原始的columns和index
    data1 = pd.concat([pd.DataFrame(data=ind_list, columns=['user_id']), data1], axis=1)  # 拼接原始的index
    data1 = data1.set_index('user_id')  # 指定矩阵的index，去除数字index
    data1.columns = col_list  # 替换矩阵的columns
    return data1


def train_model(M, K1, K2):
    print(K1)
    print(K2)
    # 初始化UBV矩阵
    col_list = M.columns.tolist()
    ind_list = M.index.tolist()
    I = M.shape[0]  # 行数
    J = M.shape[1]  # 列数
    U = np.random.rand(I, K1)
    B = np.random.rand(K1, K2)
    V = np.random.rand(J, K2)  # / (100 ** 0.5)
    T1 = np.linalg.norm(M - np.dot(U, B).dot(V.T))
    for i in range(epochs):
        V = V * np.sqrt(np.dot(M.T, U).dot(B) / np.dot(V, V.T).dot(M.T).dot(U).dot(B))
        U = U * np.sqrt(np.dot(M, V).dot(B.T) / np.dot(U, U.T).dot(M).dot(V).dot(B.T))
        B = B * np.sqrt(np.dot(U.T, M).dot(V) / np.dot(U.T, U).dot(B).dot(V.T).dot(V))
        T2 = np.linalg.norm(M - np.dot(U, B).dot(V.T))
        print('Loss:', T2)
        abs = np.abs(T1 - T2)
        if abs < end:
            print("接近收敛 退出！")
            break
        if i >= epochs - 1:
            print("达到最大次数！退出！")
        T1 = T2
    U = pd.DataFrame(U)
    B = pd.DataFrame(B)
    V = pd.DataFrame(V)
    U = pd.concat([pd.DataFrame(data=ind_list, columns=['user_id']), U], axis=1)  # 拼接U的user_id
    U = U.set_index('user_id')

    V = pd.concat([pd.DataFrame(data=col_list, columns=['item_id']), V], axis=1)  # 拼接V的item_id
    V = V.set_index('item_id')

    return U, B, V


def test_model(file_name, U, B, V, K2):
    mse_t = 0.0
    mae_t = 0.0
    num = 0
    with open(file_name, 'r') as f:  # 格式是：item_id ,score, user_id
        for index, line in enumerate(f):
            if index == 0:
                continue
            u, item, r = line.strip('\r\n').split(',')
            # print(index)
            # print(U.loc[u])
            num = index + 1
            ub = []
            for j in range(K2):
                temp = np.dot(U.loc[u], B[j])
                ub.append(temp)
            ubv = np.dot(ub, V.loc[item])

            d = float(r) - ubv
            mse_t = mse_t + pow(d, 2)
            mae_t = mae_t + np.absolute(d)
        mse = mse_t / num
        mae = mae_t / num
        print("验证集的MSE:%.5f" % mse)
    f.close()
    return mse, mae


def get_best_pama(errors):
    sort_error = errors.sort_values(by=["mse"], axis=0, ascending=True)  # 对mse列，按升序排列
    first = sort_error.iloc[0]
    print("最终的最优参数:", first)
    return first


def save_final_testerror(para, mse, mae):
    path = "../data/ubv/final_error.txt"
    f = open(path, "a", encoding='utf-8')
    f.write("最优参数：" + str(para) + "\n")
    f.write("最终mse：" + str(mse) + "\n")
    f.write("最终mae:" + str(mae) + "\n")
    f.close()


if __name__ == "__main__":
    train_name1 = "data/half/train_half1_table.csv"
    train_name2 = "../data/half/train_half2_table.csv"
    test_name1 = "../data/half/test_half1.csv"
    test_name2 = "../data/half/test_half2.csv"  # 二折交叉验证文件路径

    save_error_file = "../data/ubv/ubv_errors.csv"  # 存二折交叉误差路径

    train_data_8 = "../data/8_2/train_table.csv"  # 全部数据的80%用来训练
    test_file_2 = "../data/8_2/test.csv"  # 全部数据的20%用来测试
    train_data_all = "../data/原始数据/Office_table.csv"  # 全部数据用来训练

    list = ['K1', 'K2', 'mse', 'mae']
    errors = pd.DataFrame(columns=list)
    epochs = 1000  # 最大迭代次数
    end = 1e-3
    K1 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # 特征数1
    K2 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # 特征数2
    # K2 = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]       #特征数2
    # K1 = [10]         #特征数1
    # K2 = [10]
    train_data1 = fill_data(train_name1)
    # train_data2 = fill_data(train_name2)

    U1, B1, V1 = train_model(train_data1, 20, 40)
    # for w in range(len(K1)):
    #     for y in range(len(K2)):
    #         U1, B1, V1 = train_model(train_data1, K1[w], K2[y])  # 二折交叉第一折
    #         mse1, mae1 = test_model(test_name1, U1, B1, V1, K2[y])
    #         U2, B2, V2 = train_model(train_data2, K1[w], K2[y])  # 二折交叉第二折
    #         mse2, mae2 = test_model(test_name2, U2, B2, V2, K2[y])
    #         mse = (mse1 + mse2) / 2
    #         mae = (mae1 + mae2) / 2
    #         list = pd.DataFrame({'K1': K1[w],
    #                              'K2': K2[y],
    #                              'mse': mse,
    #                              'mae': mae},
    #                             index=[1])
    #         errors = errors.append(list, sort=False, ignore_index=True)  # 存超参数和误差
    #     errors.to_csv(save_error_file)  # 到这里已经完成了二折交叉，下一步选出最优参数，进行最终训练，得到p，q矩阵
    # print(errors)
    # para = get_best_pama(errors)  # 获取最优参数
    #
    # U_t, B_t, V_t = train_model(fill_data(train_data_8), para["K1"], para["K2"])  # 80%数据训练
    # mse_t, mae_t = test_model(test_file_2, U_t, B_t, V_t, para["K2"])  # 20%数据测试
    # save_final_testerror(para, mse_t, mae_t)  # 将最终的误差保存
    # print("测试的mse:", mse_t)
    # print("测试的mae：", mae_t)
    #
    # U_f, B_f, V_f = train_model(fill_data(train_data_all), para["K1"], para["K2"])  # 二折交叉第一折
    # U = pd.DataFrame(U_f)
    # B = pd.DataFrame(B_f)
    # V = pd.DataFrame(V_f)
    # U.to_csv('../data/ubv/U_ubv.csv')
    # B.to_csv('../data/ubv/B_ubv.csv')
    # V.to_csv('../data/ubv/V_ubv.csv')
