import numpy as np


def calculate_loss(r_ui, v_u, v_i, lamda):
    e_ui = r_ui - np.dot(v_u, v_i)
    loss = 0.5 + np.square(e_ui) + 0.5 * lamda * np.square(np.linalg.norm(v_u)) + 0.5 * lamda * np.square(np.linalg.norm(v_i))
    return loss, np.abs(e_ui)
