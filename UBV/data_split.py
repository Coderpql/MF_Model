import pandas as pd
import numpy as np


# 创建训练、测试数据
def create_train_test_data(seg_rate, path):
    data = pd.read_csv('data/data_lite/target/0_low.csv')
    print(data)


if __name__ == '__main__':
    rate = 0.9
    path = ''
    create_train_test_data(rate, path)
