import pandas as pd
import numpy as np


# ITCF
class ITCF(object):
    '''
    ITCF
    Reference: Interaction-Rich Transfer Learning for Collaborative Filtering with Heterogeneous User Feedback
    function:
        __init__(): init the model
        fit(): train
        predict(): predict
        test(): test
    :returns
        a ITCF model
    '''

    def __init__(self, args, dataset_target, dataset_source, is_columns=True):
        '''
        Init the TMF model.
        '''
        super(ITCF, self).__init__()
        # init the parameters
        self.args = args
        self.lambd = self.args['lambd']
        self.row = self.args['row']
        self.alpha = self.args['alpha']
        self.beta = self.args['beta']
        self.d = self.args['d']
        self.EPOCH = self.args['EPOCH']
        self.LR = self.args['LR']

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

        # init the latent vector
        print('init the latent vector')
        self.U = dict(zip(self.users,
                          np.random.rand(len(self.users), self.d).astype(np.float32)))
        self.W = dict(zip(self.users,
                          np.random.rand(len(self.users), self.d).astype(np.float32)))
        self.V = dict(zip(self.items,
                          np.random.rand(len(self.items), self.d).astype(np.float32)))

        # init the bias vector
        print('init the bias vector')
        self.mu = pd.DataFrame.mean(self.dataset_target[self.columns[2]], axis=0)
        self.ratings_user = self.dataset_target.groupby(self.columns[0]).agg(['mean'])[[self.columns[2]]]
        self.ratings_item = self.dataset_target.groupby(self.columns[1]).agg(['mean'])[[self.columns[2]]]
        self.B_U = dict(zip(self.ratings_user.index,
                            self.ratings_user[self.columns[2]]['mean'].astype(np.float32) - self.mu))
        self.B_I = dict(zip(self.ratings_item.index,
                            self.ratings_item[self.columns[2]]['mean'].astype(np.float32) - self.mu))
        flag = 0

    def fit(self, path_log, data_test):
        '''
        Train the TMF model.
        :return
        '''
        # Add flag to the data : {target: 1, source: 0}
        # Randomly reindex the data
        self.dataset_target['flag'] = np.ones((len(self.dataset_target.index), 1))
        self.dataset_source['flag'] = np.zeros((len(self.dataset_source.index), 1))
        data = pd.concat([self.dataset_target, self.dataset_source], ignore_index=True)
        data = data.reindex(np.random.permutation(data.index))
        l_LOSS_train = []
        l_MAE_train = []
        l_RMSE_train = []
        l_MAE_test = []
        l_RMSE_test = []
        for epoch in range(self.EPOCH):
            LOSS = 0
            mae = 0
            rmse = 0
            test_mae = 0
            test_rmse = 0
            for user, item, score, flag in data.itertuples(index=False):
                if flag:
                    v_u = self.U[user]
                    v_i = self.V[item]
                    b_u = self.B_U[user]
                    b_i = self.B_I[item]
                    v_w = self.W[user]

                    # predict r_ui
                    r_ui = np.dot(v_u, v_i) + b_u + b_i + self.mu
                    # loss of the target
                    e_ui = score - r_ui
                    mae += np.abs(e_ui)
                    rmse += np.square(e_ui)
                    l_ui = 0.5 * np.square(e_ui) + \
                           0.5 * self.alpha * (np.square(np.linalg.norm(v_u)) + np.square(np.linalg.norm(v_i))) + \
                           0.5 * self.beta * (np.square(b_u) + np.square(b_i))
                    LOSS += l_ui
                    # update the parameters
                    self.mu -= -self.LR * e_ui
                    b_u -= self.LR * (-e_ui + self.beta * b_u)
                    b_i -= self.LR * (-e_ui + self.beta * b_i)
                    v_u -= self.LR * (-e_ui * v_i + self.alpha * v_u)
                    v_i -= self.LR * (-e_ui * (self.row * v_u + (1 - self.row) * v_w) + self.alpha * v_i)
                    self.B_U[user] = b_u
                    self.B_I[item] = b_i
                    self.U[user] = v_u
                    self.V[item] = v_i
                else:
                    # get v
                    v_u = self.U[user]
                    v_i = self.V[item]
                    v_w = self.W[user]
                    # predict r_ui
                    r_ui_ = np.dot(v_w, v_i)
                    # loss of the source
                    e_ui_ = score - r_ui_
                    l_ui_ = 0.5 * np.square(e_ui_) + 0.5 * self.alpha * (np.square(np.linalg.norm(v_w)) +
                                                                         np.square(np.linalg.norm(v_i)))
                    # print(l_ui_)
                    LOSS += self.lambd * l_ui_
                    # Update the parameter
                    v_w -= self.LR * self.lambd * (-e_ui_ * v_i + self.alpha * v_w)
                    v_i -= self.LR * self.lambd * (-e_ui_ * (self.row * v_w + (1 - self.row) * v_u) +
                                                   self.alpha * v_i)
                    self.W[user] = v_w
                    self.V[item] = v_i
            self.LR *= 0.9

            # metric
            mae /= len(self.dataset_target.index)
            rmse = np.sqrt(rmse / len(self.dataset_target.index))
            test_mae, test_rmse = self.test(data_test)
            l_LOSS_train.append(LOSS)
            l_MAE_train.append(mae)
            l_RMSE_train.append(rmse)
            l_MAE_test.append(test_mae)
            l_RMSE_test.append(test_rmse)

            # Print log
            print('Epoch [{}/{}], loss: {:.4f}, train_mae: {:.4f}, train_rmse: {:.4f}, test_mae: {:.4f}, test_rmse: {:.4f}'
                  .format(epoch, self.EPOCH, LOSS, mae, rmse, test_mae, test_rmse))
        # Save log
        np.save(path_log + 'loss_train.npy', l_LOSS_train)
        np.save(path_log + 'mae_train.npy', l_MAE_train)
        np.save(path_log + 'rmse_train.npy', l_RMSE_train)
        np.save(path_log + 'mae_test.npy', l_MAE_test)
        np.save(path_log + 'rmse_test.npy', l_RMSE_test)
        return

    def predict(self, u, i):
        '''
        Predict the score between user u and item i.
        :parameter
            u: user_id of user u
            i: item_id of item i
        :return: predicted score
        '''
        b_u = self.B_U[u]
        b_i = self.B_I[i]
        v_u = self.U[u]
        v_i = self.V[i]

        r_ui = np.dot(v_u, v_i) + b_u + b_i + self.mu

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

