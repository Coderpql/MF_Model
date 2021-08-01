import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import torch

config = {
    "font.family": 'Times New Roman',  # 设置字体类型
}
rcParams.update(config)


def CreatePic():
    # 通用设置
    matplotlib.rc('axes', facecolor='white')
    matplotlib.rc('figure', figsize=(20, 12))
    matplotlib.rc('axes', grid=False)

    # 创建图形
    plt.figure(1)

    ax1 = plt.subplot(1, 1, 1)

    plt.sca(ax1)
    # 用不同的颜色表示不同数据
    plt.plot(range(mae_train.shape[0]), mae_train, color="red", linewidth=3, linestyle="-", label='mae_train')
    plt.plot(range(mae_test.shape[0]), mae_test, color="blue", linewidth=3, linestyle="-", label='mae_test')

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.title('MAE', fontsize=40)
    plt.xlabel('Number of Iterations', fontsize=40)
    plt.ylabel('MAE', fontsize=35)
    plt.legend(loc="best", fontsize=35)
    plt.savefig('../res/iteration_fig/mae.pdf', dpi=600)
    plt.show()

    # 创建图形
    plt.figure(2)

    ax1 = plt.subplot(1, 1, 1)

    plt.sca(ax1)
    # 用不同的颜色表示不同数据
    plt.plot(range(rmse_train.shape[0]), rmse_train, color="red", linewidth=3, linestyle="-", label='rmse_train')
    plt.plot(range(rmse_test.shape[0]), rmse_test, color="blue", linewidth=3, linestyle="-", label='rmse_test')

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.title('RMSE', fontsize=40)
    plt.xlabel('Number of Iterations', fontsize=40)
    plt.ylabel('RMSE', fontsize=35)
    plt.legend(loc="best", fontsize=35)
    plt.savefig('../res/iteration_fig/rmse.pdf', dpi=600)
    plt.show()

    # 创建图形
    plt.figure(3)

    ax1 = plt.subplot(1, 1, 1)

    plt.sca(ax1)
    # 用不同的颜色表示不同数据
    plt.plot(range(loss.shape[0]), loss, color="red", linewidth=3, linestyle="-", label='train_loss')

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.title('Train Loss', fontsize=40)
    plt.xlabel('Number of Iterations', fontsize=40)
    plt.ylabel('Loss', fontsize=35)
    plt.legend(loc="best", fontsize=35)
    plt.savefig('../res/iteration_fig/loss.pdf', dpi=600)
    plt.show()


if __name__ == '__main__':
    loss = np.load('../res/log/loss_train.npy')
    mae_train = np.load('../res/log/mae_train.npy')
    mae_test = np.load('../res/log/mae_test.npy')
    rmse_train = np.load('../res/log/rmse_train.npy')
    rmse_test = np.load('../res/log/rmse_test.npy')
    CreatePic()
