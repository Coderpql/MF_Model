import pandas as pd
import numpy as np
from utils.utils import calculate_loss


# FUNK_SVD_PRO
class FUNK_SVD(object):
    '''
    FUNK_SVD_PRO
    A model to solve the num bias of sample in funk_svd.
    function:
        __init__(): init the model
        fit(): train
        predict(): predict
        test(): test
    :returns
        A FUNK_SVD_PRO model
    '''

    def __init__(self, args, dataset, i_d, i_lambda, is_columns=True):
        '''
        Init the FUNK_SVD_PRO model
        '''
        super(FUNK_SVD, self).__init__()
        # init the parameters
        self.args = args
        self.alpha = self.args['alpha']
        self.lamda = self.args['lambda'][i_lambda]
        self.d = self.args['d'][i_d]
        self.EPOCH = self.args['EPOCH']
        self.columns = self.args['columns']

        # get the list of user and item
        self.dataset = dataset
        if not is_columns:
            self.dataset.columns = self.columns
        self.users = list(set(list(self.dataset[self.columns[0]])))
        self.items = list(set(list(self.dataset[self.columns[1]])))

        # init P,Q
        self.P = dict(zip(self.users,
                          np.random.rand(len(self.users), self.d).astype(np.float32)))
        self.Q = dict(zip(self.items,
                          np.random.rand(len(self.users), self.d).astype(np.float32)))

    def fit(self):
        '''
        Train the FUNK_SVD_PRO model
        :return:
        '''
        loss_old = 100000000
        for epoch in range(self.EPOCH):
            for uid, iid, r_ui in self.dataset.itertuples(index=False):
                v_u = self.P[uid]
                v_i = self.Q[iid]

                e_ui = r_ui - np.dot(v_u, v_i)
                v_u += self.alpha * (e_ui * v_i - self.lamda * v_u)
                v_i += self.alpha * (e_ui * v_i - self.lamda * v_i)

                self.P[uid] = v_u
                self.Q[iid] = v_i

            # calculate loss
            L = []
            MAE = []

            for uid, iid, r_ui in self.dataset.itertuples(index=False):
                v_u = self.P[uid]
                v_i = self.Q[iid]
                l, m = calculate_loss(r_ui=r_ui, v_u=v_u, v_i=v_i, lamda=self.lamda)
                L.append(l)
                MAE.append(m)

            loss = np.sum(np.array(L))
            mae = np.sum(np.array(MAE)) / len(MAE)
            loss_c = np.abs(loss - loss_old)
            loss_old = loss
            if loss_c < 0.001:
                break

            # print log
            print('Epoch [{}/{}], d: {}, lambda: {}, loss: {:.4f}, mae: {:.4f}'.format(epoch, self.EPOCH, self.d,
                                                                                       self.lamda, loss, mae))

    def predict(self, u, i):
        '''
        Predict the score between user u and item i.
        :param u: user_id of user u
        :param i: item_id of item i
        :return: predicted score
        '''
        v_u = self.P[u]
        v_i = self.Q[i]

        r_ui = np.dot(v_u, v_i)

        return r_ui

    def test(self, data_test):
        '''
        Use the test dataset to test the performance of the TMF model.
        :parameter
            dataset: test dataset_target
        :returns
            mae: mean average error
            rmse: root mean squared error
        '''
        mae = 0
        rmse = 0
        for user, item, score in data_test.itertuples(index=False):
            r_ui = self.predict(user, item)
            e_ui = score - r_ui
            mae += np.abs(e_ui)
            rmse += np.square(e_ui)
        mae /= len(data_test.index)
        rmse = np.sqrt(rmse / len(data_test.index))

        return mae, rmse
