import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.family": 'Times New Roman',  # 设置字体类型
}
rcParams.update(config)


def CreatePic(loss, name):
    # 通用设置
    matplotlib.rc('axes', facecolor='white')
    matplotlib.rc('figure', figsize=(20, 12))
    matplotlib.rc('axes', grid=False)

    # 创建图形
    plt.figure(1)

    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 3, 4)
    ax5 = plt.subplot(2, 3, 5)
    ax6 = plt.subplot(2, 3, 6)

    i = 0

    plt.sca(ax1)
    # 用不同的颜色表示不同数据
    plt.plot(range(loss[i].shape[0]), loss[i], color="blue", linewidth=3, linestyle="-")
    plt.scatter([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                loss[i][[0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]], s=150, color='blue', marker='o')

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.title('D' + str(i + 1), fontsize=40)
    plt.xlabel('Number of Iterations', fontsize=40)
    plt.ylabel('Loss', fontsize=35)
    i += 1

    plt.sca(ax2)
    # 用不同的颜色表示不同数据
    plt.plot(range(loss[i].shape[0]), loss[i], color="blue", linewidth=3, linestyle="-")
    plt.scatter([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                loss[i][[0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]], s=150, color='blue', marker='o')

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.title('D' + str(i + 1), fontsize=40)
    plt.xlabel('Number of Iterations', fontsize=40)
    plt.ylabel('Loss', fontsize=35)
    i += 1

    plt.sca(ax3)
    # 用不同的颜色表示不同数据
    plt.plot(range(loss[i].shape[0]), loss[i], color="blue", linewidth=3, linestyle="-")
    plt.scatter([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                loss[i][[0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]], s=150, color='blue', marker='o')

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.title('D' + str(i + 1), fontsize=40)
    plt.xlabel('Number of Iterations', fontsize=40)
    plt.ylabel('Loss', fontsize=35)
    i += 1

    plt.sca(ax4)
    # 用不同的颜色表示不同数据
    plt.plot(range(loss[i].shape[0]), loss[i], color="blue", linewidth=3, linestyle="-")
    plt.scatter([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                loss[i][[0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]], s=150, color='blue', marker='o')

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.title('D' + str(i + 1), fontsize=40)
    plt.xlabel('Number of Iterations', fontsize=40)
    plt.ylabel('Loss', fontsize=35)
    i += 1

    plt.sca(ax5)
    # 用不同的颜色表示不同数据
    plt.plot(range(loss[i].shape[0]), loss[i], color="blue", linewidth=3, linestyle="-")
    plt.scatter([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                loss[i][[0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]], s=150, color='blue', marker='o')

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.title('D' + str(i + 1), fontsize=40)
    plt.xlabel('Number of Iterations', fontsize=40)
    plt.ylabel('Loss', fontsize=35)
    i += 1

    plt.sca(ax6)
    # 用不同的颜色表示不同数据
    plt.plot(range(loss[i].shape[0]), loss[i], color="blue", linewidth=3, linestyle="-")
    plt.scatter([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                loss[i][[0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]], s=150, color='blue', marker='o')

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.title('D' + str(i + 1), fontsize=40)
    plt.xlabel('Number of Iterations', fontsize=40)
    plt.ylabel('Loss', fontsize=35)
    plt.tight_layout()
    i += 1

    plt.savefig(name, dpi=600)
    plt.show()
