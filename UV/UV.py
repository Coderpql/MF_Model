'''
LFM Model
'''
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# 超参数
Lr = 0.01
Epoch = 2000


# 评分预测    1-5
class LFM(object):

    def __init__(self, alpha, lamda, number_LatentFactors=10, number_epochs=10,
                 columns=None):
        if columns is None:
            columns = ['user_id', 'items_id', 'score']
        self.alpha = alpha  # 学习率
        self.lamda = lamda  # 正则项系数
        self.number_LatentFactors = number_LatentFactors  # 隐式类别数量
        self.number_epochs = number_epochs  # 最大迭代次数
        self.columns = columns

    def fit(self, dataset):
        '''
        fit dataset
        :param dataset: uid, iid, rating
        :return:
        '''

        self.dataset = pd.DataFrame(dataset)

        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]

        self.globalMean = self.dataset[self.columns[2]].mean()

        self.P, self.Q = self.sgd()

    def _init_matrix(self):
        '''
        初始化P和Q矩阵，同时为设置0，1之间的随机值作为初始值
        :return:
        '''
        # User-LF
        P = dict(zip(
            self.users_ratings.index,
            np.random.rand(len(self.users_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        # Item-LF
        Q = dict(zip(
            self.items_ratings.index,
            np.random.rand(len(self.items_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        return P, Q

    def sgd(self):
        '''
        使用随机梯度下降，优化结果
        :return:
        '''
        P, Q = self._init_matrix()
        loss_old = 100000000000000
        for i in range(self.number_epochs):
            error_list = []
            for uid, iid, r_ui in self.dataset.itertuples(index=False):
                # User-LF P
                # Item-LF Q
                v_pu = P[uid]  # 用户向量
                v_qi = Q[iid]  # 物品向量
                err = np.float32(r_ui - np.dot(v_pu, v_qi))

                v_pu += self.alpha * (err * v_qi - self.lamda * v_pu)
                v_qi += self.alpha * (err * v_pu - self.lamda * v_qi)

                P[uid] = v_pu
                Q[iid] = v_qi

                error_list.append(err ** 2)
            loss = np.sum(error_list)
            loss_c = loss_old - loss
            loss_old = loss
            # print('Epoch: %d, Loss: %f' % (i, np.sqrt(np.mean(error_list))))
            if loss_c < 0.001:
                break
        return P, Q

    def predict(self, uid, iid):
        # 如果uid或iid不在，我们使用全剧平均分作为预测结果返回
        if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
            return self.globalMean

        p_u = self.P[uid]
        q_i = self.Q[iid]

        return np.dot(p_u, q_i)

    def test(self, testset):
        '''预测测试集数据'''
        l_score_real = []
        l_score_pred = []
        for uid, iid, real_rating in testset.itertuples(index=False):
            try:
                pred_rating = self.predict(uid, iid)
            except Exception as e:
                print(e)
            else:
                l_score_pred.append(pred_rating)
                l_score_real.append(real_rating)
        l_score_real = np.array(l_score_real).reshape(-1, 1)
        l_score_pred = np.array(l_score_pred).reshape(-1, 1)
        mae = np.abs(l_score_pred - l_score_real).mean()
        rmse = np.sqrt(mean_squared_error(l_score_real, l_score_pred))

        return l_score_real, l_score_pred, mae, rmse


# K折交叉验证
def verify_k_fold(k_fold):
    train_data = None
    test_data = None

    K = [5, 10, 15, 20, 25, 30, 35, 40]
    lamdas = [0.001, 0.01, 0.1]
    for p in range(25, 45, 5):
        # 创建存储交叉验证结果的表格
        li = ['TR', 'K', 'Lamda', 'mae_k_fold']
        res = pd.DataFrame(columns=li)
        res.to_csv('res/k_fold_res/k_fold_mae_' + str(p) + '.csv', mode='a')
        for k in K:
            for lamda in lamdas:
                MAE = 0
                for i in range(k_fold):
                    # 读取验证数据
                    path_data = 'data_no_mat/target_k_fold/0_' + str(p) + '_' + str(i) + '.csv'
                    test_data = pd.read_csv(path_data)
                    n = 0
                    for j in range(k_fold):
                        # 读取训练数据
                        path_data = 'data_no_mat/target_k_fold/0_' + str(p) + '_' + str(j) + '.csv'
                        data = pd.read_csv(path_data)
                        if n == 0:
                            train_data = data
                        else:
                            train_data = train_data.append(data, sort=False)
                        n += 1

                    # 构建模型
                    lfm = LFM(Lr, lamda, k, Epoch)
                    # 模型训练
                    lfm.fit(train_data)
                    # 模型测试
                    rating_real, rating_pred, mae, rmse = lfm.test(test_data)
                    MAE += mae
                MAE /= k_fold
                # 统计交叉验证结果
                result = pd.DataFrame({'TR': [p], 'K': [k], 'Lamda': [lamda], 'mae_k_fold': [MAE]})
                res = res.append(result, sort=False, ignore_index=True)
                res.to_csv('res/k_fold_res/k_fold_mae_' + str(p) + '.csv', mode='a', header=False)
                print('TR', p, 'K:', k, 'Lamda:', lamda, 'MAE:', MAE)


def test_UV():
    rain_data = None
    test_data = None
    K = [40]
    lamdas = [0.001]
    parameter = pd.read_csv('res/uv_k_fold.csv')
    parameter.set_index(parameter['TR'], inplace=True)
    print(parameter)
    # 创建存储交叉验证结果的表格
    li = ['TR', 'K', 'Lamda', 'mae', 'rmse']
    res = pd.DataFrame(columns=li)
    res.to_csv('res/k_fold_res/mae_rmse.csv', mode='a', index=False)
    for p in range(25, 45, 5):
        # 读取参数
        lamda = parameter.loc[p, 'Lamda']
        k = parameter.loc[p, 'K']
        # 读取训练数据
        path_train = 'data_no_mat/target/0_' + str(p) + '.csv'
        train_data = pd.read_csv(path_train)
        # 读取测试数据
        path_test = 'data_no_mat/target/0_' + str(p) + '_test.csv'
        test_data = pd.read_csv(path_test)
        # 构建模型
        lfm = LFM(Lr, lamda, k, Epoch)
        # 模型训练
        lfm.fit(train_data)
        # 模型测试
        rating_real, rating_pred, mae, rmse = lfm.test(test_data)

        # 结果
        result = pd.DataFrame({'TR': [p], 'K': [k], 'Lamda': [lamda], 'mae': [mae], 'rmse': [rmse]})
        result.to_csv('res/mae_rmse.csv', mode='a', index=False, header=False)
        print('TR', p, 'K:', k, 'Lamda:', lamda, 'MAE:', mae, 'RMSE', rmse)


if __name__ == '__main__':
    K_Fold = 2
    verify_k_fold(K_Fold)
    test_UV()
