from functools import partial
import numpy as np
from multiprocessing import Pool
from queue import Queue
from utils.function import calculate_x_ui, create_rating_matrix


# CMTF
class CMTF(object):
    '''
    CMTF
    Reference: Transfer Learning to Predict Missing Ratings via Heterogeneous User Feedbacks
    function:
        __init__(): init the model
        fit(): train
        predict(): predict
        test(): test
    :returns
        a CMTF model
    '''

    def __init__(self, args, dataset_target, dataset_source, is_columns=True):
        '''
        Init the CMTF model
        :param args: parameter
        :param dataset_target: dataset of target domain
        :param dataset_source: dataset of source domain
        :param is_columns: Does the dataset include columns?
        '''
        super(CMTF, self).__init__()

        # init the parameters
        self.args = args
        self.alpha_u = self.args['alpha_u']
        self.alpha_v = self.args['alpha_v']
        self.beta = self.args['beta']
        self.d = self.args['d']
        self.lambd = self.args['lambda']
        self.gamma = 0
        self.EPOCH_UV = self.args['EPOCH_UV']
        self.EPOCH_B = self.args['EPOCH_B']
        self.process = self.args['process']

        # initial pool
        self.processPool = Pool(processes=self.process)
        self.q = Queue()

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

        # group data
        self.rating_u_T = self.dataset_target.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.rating_v_T = self.dataset_target.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        self.rating_u_S = self.dataset_source.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.rating_v_S = self.dataset_source.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]

        # init the latent vector
        print('init the latent vector')
        self.U = dict(zip(self.users,
                          np.random.rand(len(self.users), self.d).astype(np.float32)))
        self.V = dict(zip(self.items,
                          np.random.rand(len(self.items), self.d).astype(np.float32)))

        # init B via LSSVM
        self.B = None
        self.B_ = None
        self.estimate_B()

    # estimate B and B_
    def estimate_B(self):
        print('estimate_B')
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
            v_u = self.U[user].reshape(1, -1)
            v_v = self.V[item].reshape(1, -1)
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
            v_u = self.U[user].reshape(1, -1)
            v_v = self.V[item].reshape(1, -1)
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
        Train the CMTF model.
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

                # update U
                for user in self.users:
                    C_u = 0
                    b_u = 0
                    # Target
                    l_items = self.rating_u_T.loc[user, self.columns[1]][0]
                    l_ratings = self.rating_u_T.loc[user, self.columns[2]][0]
                    for i in range(len(l_items)):
                        v_v = self.V[l_items[i]].reshape(1, -1)
                        r_ui = float(l_ratings[i] - 1) / 4

                        C_u += np.dot(np.dot(np.dot(self.B, v_v.T), v_v), self.B.T) + self.alpha_u * np.eye(self.d)
                        b_u += r_ui * np.dot(v_v, self.B.T)

                    # source
                    l_items = self.rating_u_S.loc[user, self.columns[1]][0]
                    l_ratings = self.rating_u_S.loc[user, self.columns[2]][0]
                    for j in range(len(l_items)):
                        v_v = self.V[l_items[j]].reshape(1, -1)
                        r_ui_ = l_ratings[j]

                        C_u += self.lambd * np.dot(np.dot(np.dot(self.B_, v_v.T), v_v),
                                                   self.B_.T) + self.alpha_u * self.lambd * np.eye(self.d)
                        b_u += self.lambd * r_ui_ * np.dot(v_v, self.B_.T)

                    v_u = np.dot(b_u, np.linalg.inv(C_u))
                    self.U[user] = v_u

                # update V
                for item in self.items:
                    C_i = 0
                    b_i = 0
                    # Target
                    l_users = self.rating_v_T.loc[item, self.columns[0]][0]
                    l_ratings = self.rating_v_T.loc[item, self.columns[2]][0]
                    for i in range(len(l_users)):
                        v_u = self.V[l_users[i]].reshape(1, -1)
                        r_ui = float(l_ratings[i] - 1) / 4

                        C_i += np.dot(np.dot(np.dot(self.B.T, v_u.T), v_u), self.B) + self.alpha_v * np.eye(self.d)
                        b_i += r_ui * np.dot(v_u, self.B)

                    # source
                    l_users = self.rating_v_S.loc[item, self.columns[0]][0]
                    l_ratings = self.rating_v_S.loc[item, self.columns[2]][0]
                    for j in range(len(l_users)):
                        v_u = self.V[l_users[j]].reshape(1, -1)
                        r_ui_ = l_ratings[j]

                        C_i += self.lambd * np.dot(np.dot(np.dot(self.B_.T, v_u.T), v_u),
                                                   self.B_) + self.alpha_v * self.lambd * np.eye(self.d)
                        b_i += self.lambd * r_ui_ * np.dot(v_u, self.B_)

                    v_v = np.dot(b_i, np.linalg.inv(C_i))
                    self.V[item] = v_v
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
            v_u = self.U[user].reshape(1, -1)
            v_v = self.V[item].reshape(1, -1)
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
            v_u = self.U[user].reshape(1, -1)
            v_v = self.V[item].reshape(1, -1)
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
        v_u = self.U[u].reshape(1, -1)
        v_v = self.V[i].reshape(1, -1)

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
