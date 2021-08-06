from functools import partial
import pandas as pd
import numpy as np
from multiprocessing import Pool
from sklearn import clone
from utils.function import calculate_x_ui, create_rating_matrix
from scipy.sparse.linalg import svds


# CSVD
class CSVD(object):
    '''
    CSVD
    Reference: Transfer Learning to Predict Missing Ratings via Heterogeneous User Feedbacks
    function:
        __init__(): init the model
        fit(): train
        predict(): predict
        test(): test
    :returns
        a CSVD model
    '''

    def __init__(self, args, dataset_target, dataset_source, is_columns=True):
        '''
        Init the CSVD model
        :param args: parameter
        :param dataset_target: dataset of target domain
        :param dataset_source: dataset of source domain
        :param is_columns: Does the dataset include columns?
        '''
        super(CSVD, self).__init__()

        # init the parameters
        self.args = args
        self.alpha_u = self.args['alpha_u']
        self.alpha_v = self.args['alpha_v']
        self.beta = self.args['beta']
        self.d = self.args['d']
        self.lambd = self.args['lambda']
        self.EPOCH_UV = self.args['EPOCH_UV']
        self.EPOCH_B = self.args['EPOCH_B']
        self.process = self.args['process']

        # initial pool
        self.processPool = Pool(processes=self.process)

        # get the list of user and item
        print('get the list of user and item')
        self.dataset_target = dataset_target
        self.dataset_source = dataset_source
        self.columns = ['user_id', 'item_id', 'score']
        if not is_columns:
            self.dataset_target.columns = self.columns
            self.dataset_source.columns = self.columns
        self.users = list(set(list(self.dataset_target[self.columns[0]])))
        self.items = list(set(list(self.dataset_target[self.columns[1]])))

        # init the rating matrix
        self.R = create_rating_matrix(self.dataset_target).values
        self.R = (self.R - 1) / 4
        self.R[self.R < 0] = 0
        self.R_ = create_rating_matrix(self.dataset_source).values
        self.R_ = (self.R_ - 1) / 4
        self.R_[self.R_ < 0] = 0

        # init the rating mask
        self.Y = self.R.copy()
        self.Y[self.Y > 0] = 1
        self.Y_ = self.R_.copy()
        self.Y_[self.Y_ > 0] = 1

        # group data
        self.rating_u_T = self.dataset_target.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.rating_v_T = self.dataset_target.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        self.rating_u_S = self.dataset_source.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.rating_v_S = self.dataset_source.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]

        # init the latent vector
        print('init the latent vector')

        # init UV via SVD
        U, B, V = svds(self.R_, k=self.d)
        self.U = U
        self.V = V.T
        self.U_dict = dict(zip(self.users, self.U))
        self.V_dict = dict(zip(self.items, self.V))

        # init B via LSSVM
        print('estimate_B')
        self.B = None
        self.B_ = None
        self.estimate_B()

    # estimate B and B_
    def estimate_B(self):
        '''
        estimate B and B_
        :return:
        '''
        # estimate B
        X = []
        r = []
        i = 0

        self.processPool = Pool(processes=self.process)
        for user, item, score in self.dataset_target.itertuples(index=False):
            v_u = self.U_dict[user].reshape(1, -1)
            v_v = self.V_dict[item].reshape(1, -1)
            partial_func = partial(calculate_x_ui, i, v_u=v_u, v_v=v_v)
            X.append(self.processPool.apply_async(partial_func))
            r.append(float(score - 1) / 4)
            i += 1

        self.processPool.close()
        self.processPool.join()

        for i in range(len(X)):
            X[i] = X[i].get()

        X = np.array(X).reshape(-1, np.square(self.d))

        r = np.array(r).reshape((-1, 1))
        w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + self.beta * np.eye(np.square(self.d))), X.T), r)
        self.B = w.reshape((self.d, self.d), order='F')

        # estimate B_
        X_ = []
        r_ = []
        i = 0

        self.processPool = Pool(processes=self.process)
        for user, item, score in self.dataset_source.itertuples(index=False):
            v_u = self.U_dict[user].reshape(1, -1)
            v_v = self.V_dict[item].reshape(1, -1)
            partial_func = partial(calculate_x_ui, i, v_u=v_u, v_v=v_v)
            X_.append(self.processPool.apply_async(partial_func))
            r_.append(score)
            i += 1

        self.processPool.close()
        self.processPool.join()

        for i in range(len(X_)):
            X_[i] = X_[i].get()

        X_ = np.array(X_).reshape(-1, np.square(self.d))

        r_ = np.array(r_).reshape((-1, 1))
        w = np.dot(np.dot(np.linalg.inv(np.dot(X_.T, X_) + self.beta * np.eye(np.square(self.d))), X_.T), r_)
        self.B_ = w.reshape((self.d, self.d), order='F')

    def fit(self, path_log, data_test):
        '''
        Train the CSVD model.
        :param path_log: the path of the log
        :param data_test: the test dataset
        :return:
        '''
        l_LOSS_train = []
        l_MAE_train = []
        l_RMSE_train = []
        l_MAE_test = []
        l_RMSE_test = []
        loss_old = 1000000000
        for epoch_b in range(self.EPOCH_B):
            # loss of target
            LOSS, mae, rmse = self.get_loss()

            # metric
            test_mae, test_rmse = self.test(data_test)
            l_LOSS_train.append(LOSS)
            l_MAE_train.append(mae)
            l_RMSE_train.append(rmse)
            l_MAE_test.append(test_mae)
            l_RMSE_test.append(test_rmse)

            # Print log
            print(
                'Epoch [{}/{}], loss: {:.4f}, train_mae: {:.4f}, train_rmse: {:.4f}, test_mae: {:.4f}, test_rmse: {:.4f}'
                    .format(epoch_b, self.EPOCH_B, LOSS, mae, rmse, test_mae, test_rmse))

            loss_c = np.abs(loss_old - LOSS)
            loss_old = LOSS
            if loss_c <= 0.01:
                print('Model Convergence!')
                break

            # update UV
            loss_old_uv = 1000000000
            for epoch_uv in range(self.EPOCH_UV):
                LOSS_uv, mae_uv, rmse_uv = self.get_loss()
                test_mae_uv, test_rmse_uv = self.test(data_test)

                # Print log
                print(
                    'Epoch_UV [{}/{}], loss: {:.4f}, train_mae: {:.4f}, train_rmse: {:.4f}, test_mae: {:.4f}, test_rmse: {:.4f}'
                        .format(epoch_uv, self.EPOCH_UV, LOSS_uv, mae_uv, rmse_uv, test_mae_uv, test_rmse_uv))

                loss_c = np.abs(loss_old_uv - LOSS_uv)
                loss_old_uv = LOSS_uv
                if loss_c <= 0.01:
                    print('UV Convergence!')
                    break

                # calculate gradient of U
                g_U = np.dot(np.dot((self.Y * (np.dot(np.dot(self.U, self.B), self.V.T) - self.R)), self.V), self.B.T)
                + self.lambd * np.dot(np.dot((self.Y_ * (np.dot(np.dot(self.U, self.B_), self.V.T) - self.R_)), self.V),
                                      self.B_.T)

                delta_U = np.dot((np.eye(len(self.users)) - np.dot(self.U, self.U.T)), g_U)

                # calculate gradient of V
                g_V = np.dot(np.dot((self.Y * (np.dot(np.dot(self.U, self.B), self.V.T) - self.R)), self.U), self.B.T)
                + self.lambd * np.dot(np.dot((self.Y_ * (np.dot(np.dot(self.U, self.B_), self.V.T) - self.R_)), self.U),
                                      self.B_.T)
                delta_V = np.dot((np.eye(len(self.users)) - np.dot(self.V, self.V.T)), g_V)

                # calculate gamma
                t_1 = self.Y * (self.R - np.dot(np.dot(self.U, self.B), self.V.T))
                t_1_ = self.Y_ * (self.R_ - np.dot(np.dot(self.U, self.B_), self.V.T))
                t_2 = self.Y * np.dot(np.dot(delta_U, self.B), self.V.T)
                t_2_ = self.Y_ * np.dot(np.dot(delta_U, self.B_), self.V.T)
                gamma = float(-np.trace(np.dot(t_1.T, t_2)) - self.lambd * np.trace(np.dot(t_1_.T, t_2_))) / \
                        (np.trace(np.dot(t_2.T, t_2)) + self.lambd * np.trace(np.dot(t_2_.T, t_2_)))

                # update U
                self.U -= gamma * delta_U
                self.U_dict = dict(zip(self.users, self.U))

                # update V
                self.V -= gamma * delta_V
                self.V_dict = dict(zip(self.items, self.V))
            # update B
            self.estimate_B()

        # Save log
        np.save(path_log + 'loss_train.npy', l_LOSS_train)
        np.save(path_log + 'mae_train.npy', l_MAE_train)
        np.save(path_log + 'rmse_train.npy', l_RMSE_train)
        np.save(path_log + 'mae_test.npy', l_MAE_test)
        np.save(path_log + 'rmse_test.npy', l_RMSE_test)

    def get_loss(self):
        '''
        Calculate the loss of the model.
        :return: LOSS, mae, rmse
        '''
        LOSS = 0
        loss_T = 0
        loss_S = 0
        mae = 0
        rmse = 0
        for user, item, score in self.dataset_target.itertuples(index=False):
            v_u = self.U_dict[user].reshape(1, -1)
            v_v = self.V_dict[item].reshape(1, -1)
            r_ui = score
            e_ui = r_ui - np.dot(np.dot(v_u, self.B), v_v.T)[0, 0]
            err = r_ui - (np.dot(np.dot(v_u, self.B), v_v.T)[0, 0] * 4 + 1)
            mae += np.abs(err)
            rmse += np.square(err)
            loss_T += 0.5 * np.square(e_ui) + 0.5 * self.alpha_u * np.square(np.linalg.norm(v_u)) + \
                      0.5 * self.alpha_v * np.square(np.linalg.norm(v_v))
        loss_T += 0.5 * self.beta * np.square(np.linalg.norm(self.B))
        # loss of source
        for user, item, score in self.dataset_source.itertuples(index=False):
            v_u = self.U_dict[user].reshape(1, -1)
            v_v = self.V_dict[item].reshape(1, -1)
            r_ui_ = score
            err_ = r_ui_ - np.dot(np.dot(v_u, self.B_), v_v.T)[0, 0]
            loss_S += 0.5 * np.square(err_) + 0.5 * self.alpha_u * np.square(np.linalg.norm(v_u)) + \
                      0.5 * self.alpha_v * np.square(np.linalg.norm(v_v))
        loss_S += 0.5 * self.beta * np.square(np.linalg.norm(self.B_))
        LOSS += loss_T + self.lambd * loss_S

        mae /= len(self.dataset_target.index)
        rmse = np.sqrt(rmse / len(self.dataset_target.index))

        return LOSS, mae, rmse

    def predict(self, u, i):
        '''
        Predict the score between user u and item i.
        :parameter
            u: user_id of user u
            i: item_id of item i
        :return: predicted score
        '''
        v_u = self.U_dict[u].reshape(1, -1)
        v_v = self.V_dict[i].reshape(1, -1)

        r_ui = np.dot(np.dot(v_u, self.B), v_v.T)[0, 0] * 4 + 1

        return r_ui

    def test(self, dataset):
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
        for user, item, score in dataset.itertuples(index=False):
            r_ui = self.predict(user, item)
            e_ui = score - r_ui
            mae += np.abs(e_ui)
            rmse += np.square(e_ui)

        mae /= len(dataset.index)
        rmse = np.sqrt(rmse / len(dataset.index))

        return mae, rmse
