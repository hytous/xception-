from matplotlib import pyplot as plt

Loss_list = []  # 存储每次epoch损失值


# loss曲线
def draw_loss(Loss_list, epoch):
    # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    plt.cla()
    x1 = range(1, epoch + 1)
    print(x1)
    y1 = Loss_list
    print(y1)
    plt.title('Train loss vs. epoches', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoches', fontsize=20)
    plt.ylabel('Train loss', fontsize=20)
    plt.grid()
    # plt.savefig("./lossAndacc/Train_loss.png")
    # plt.savefig("./lossAndacc/Train_loss.png")
    plt.show()


# acc和loss曲线
def draw_fig(datalist, name, epoch):
    # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    x1 = range(1, epoch + 1)
    print(x1)
    y1 = datalist
    if name == "loss":
        plt.cla()  # 清除之前的
        plt.title('Train loss vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train loss', fontsize=20)
        plt.grid()
        # plt.savefig("./lossAndacc/Train_loss.png")
        plt.show()
    elif name == "acc":
        plt.cla()
        plt.title('Train accuracy vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train accuracy', fontsize=20)
        plt.grid()
        # plt.savefig("./lossAndacc/Train _accuracy.png")
        plt.show()
