import pandas as pd
import numpy as np


# TMF
class TMF(object):
    '''
    TMF
    Reference: Mixed factorization for collaborative recommendation with heterogeneous explicit feedbacks
    function:
        __init__(): init the model
        fit(): train
        predict(): predict
        test(): test
    :returns
        a TMF model
    '''

    def __init__(self, args, dataset_target, dataset_source, is_columns=True):
        '''
        Init the TMF model.
        '''
        super(TMF, self).__init__()
        # init the parameters
        self.args = args
        self.lambd = self.args['lambd']
        self.row = self.args['row']
        self.alpha = self.args['alpha']
        self.beta = self.args['beta']
        self.d = self.args['d']
        self.w_p = self.args['w_p']
        self.w_n = self.args['w_n']
        self.delta_p = self.args['delta_p']
        self.delta_n = self.args['delta_n']
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

        # init the like- and dislike-vector
        self.ratings_user_ = self.dataset_source.groupby(self.columns[0]).agg([list])[[self.columns[1],
                                                                                       self.columns[2]]]
        flag = 0
        vector_like = None
        vector_dislike = None
        self.U_like = dict()
        self.U_dislike = dict()
        for u in self.ratings_user_.index:
            # Get the items and scores which are associated with user u
            items_u = self.ratings_user_.loc[u, self.columns[1]]
            scores_u = self.ratings_user_.loc[u, self.columns[2]]
            # Get the position of like- and dislike-items
            position_likes = [i for i, x in enumerate(scores_u) if x == 1]
            position_dislikes = [i for i, x in enumerate(scores_u) if x == 0]
            # Get the like- and dislike-items
            items_likes = [items_u[i] for i in position_likes]
            self.U_like[u] = items_likes
            items_dislikes = [items_u[i] for i in position_dislikes]
            self.U_dislike[u] = items_dislikes
            # calculate the like- and dislike-vector
            v_like = np.zeros((1, self.d))
            for i in items_likes:
                v_like += self.V[i]
            if len(items_likes) != 0:
                v_like /= np.sqrt(len(items_likes))
            v_dislike = np.zeros((1, self.d))
            for i in items_dislikes:
                v_dislike += self.V[i]
            if len(items_dislikes) != 0:
                v_dislike /= np.sqrt(len(items_dislikes))
            if flag == 0:
                vector_like = v_like
                vector_dislike = v_dislike
                flag += 1
            else:
                vector_like = np.vstack((vector_like, v_like))
                vector_dislike = np.vstack((vector_dislike, v_dislike))
                flag += 1

        self.P = dict(zip(self.ratings_user_.index, vector_like))
        self.N = dict(zip(self.ratings_user_.index, vector_dislike))

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
                    p_u = self.P[user]
                    n_u = self.N[user]
                    v_w = self.W[user]

                    # like-dislike-feature
                    Au = self.delta_p * self.w_p * p_u + self.delta_n * self.w_n * n_u
                    # predict r_ui
                    r_ui = np.dot(v_u, v_i) + np.dot(Au, v_i) + b_u + b_i + self.mu
                    # loss of the target
                    reg_p = 0
                    for p in self.U_like[user]:
                        reg_p += np.square(np.linalg.norm(self.V[p]))
                    reg_n = 0
                    for n in self.U_dislike[user]:
                        reg_n += np.square(np.linalg.norm(self.V[n]))
                    e_ui = score - r_ui
                    mae += np.abs(e_ui)
                    rmse += np.square(e_ui)
                    l_ui = 0.5 * np.square(e_ui) + \
                           0.5 * self.alpha * (np.square(np.linalg.norm(v_u)) + np.square(np.linalg.norm(v_i))) + \
                           0.5 * self.beta * (np.square(b_u) + np.square(b_i)) + \
                           0.5 * self.alpha * self.delta_p * reg_p + \
                           0.5 * self.alpha * self.delta_n * reg_n
                    LOSS += l_ui
                    # update the parameters
                    self.mu -= -self.LR * e_ui
                    b_u -= self.LR * (-e_ui + self.beta * b_u)
                    b_i -= self.LR * (-e_ui + self.beta * b_i)
                    v_u -= self.LR * (-e_ui * v_i + self.alpha * v_u)
                    v_i -= self.LR * (-e_ui * (self.row * v_u + (1 - self.row) * v_w + Au) + self.alpha * v_i)
                    for p in self.U_like[user]:
                        self.V[p] -= self.LR * self.delta_p * (-e_ui * self.w_p * (1 / float(np.sqrt(len(self.U_like))))
                                                               + self.alpha * self.V[p])
                    for n in self.U_dislike[user]:
                        self.V[n] -= self.LR * self.delta_n * (
                                -e_ui * self.w_n * (1 / float(np.sqrt(len(self.U_dislike))))
                                + self.alpha * self.V[n])
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
        p_u = self.P[u]
        n_u = self.N[i]
        b_u = self.B_U[u]
        b_i = self.B_I[i]
        v_u = self.U[u]
        v_i = self.V[i]
        Au = self.delta_p * self.w_p * p_u + self.delta_n * self.w_n * n_u

        r_ui = np.dot(v_u, v_i) + np.dot(Au, v_i) + b_u + b_i + self.mu

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

