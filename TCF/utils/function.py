import numpy as np
import pandas as pd


def calculate_x_ui(i, v_u, v_v):
    x_ui = np.dot(v_u.T, v_v).reshape((1, -1), order='F')  # shape: (1, d^2) --> (, d^2)
    return x_ui


def calculate_Cu_bu(B, v_v, alpha_u, r_ui):
    C_u = np.dot(np.dot(np.dot(B, v_v.T), v_v), B.T) + alpha_u * 1
    b_u = r_ui * np.dot(v_v, B.T)
    return C_u, b_u


def create_rating_matrix(data):
    table_select = pd.pivot_table(data, index=['user_id'], columns=['item_id'], values=['score'])
    table_select.columns = table_select.columns.droplevel()
    table_select = table_select.reset_index()
    table_select = pd.concat([pd.DataFrame(data=table_select.index.tolist(),
                                           columns=[table_select.index.name],
                                           index=table_select.index.tolist()), table_select], axis=1)
    col_list = table_select.columns.tolist()
    col_list.remove(None)
    table_select = table_select.loc[:, col_list]
    table_select = table_select.set_index('user_id')
    table_select = table_select.fillna(0.0)
    return table_select.astype('float')
