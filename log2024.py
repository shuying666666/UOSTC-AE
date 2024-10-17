# #用一张图显示多条tensorboard线，如果不用tensorboard这个也就废弃了
#
# from torch.utils import tensorboard
# import numpy as np
#
# writer = {
#     'loss': tensorboard.SummaryWriter("./drive/MyDrive/logs/loss"),  # 必须要不同的writer
#     'acc': tensorboard.SummaryWriter("./drive/MyDrive/logs/acc"),
#     'lr': tensorboard.SummaryWriter("./drive/MyDrive/logs/lr")
# }
#
# data = np.random.random((3, 10))  # 生成模拟数据，生成一个形状为 (3, 10) 的二维数组，其中的每个元素都是 [0, 1) 范围内的随机数
#
# loss_data = data[0]
# acc_data = data[1]
# lr_data = data[2]
#
# for i in range(10):
#     writer['loss'].add_scalar("data", loss_data[i], i)  # 要想显示在一张图 表格名字要一样！！
#     writer['acc'].add_scalar("data", acc_data[i], i)
#     writer['lr'].add_scalar("data", lr_data[i], i)
#
# writer['loss'].close()
# writer['acc'].close()
# writer['lr'].close()
# exit()
# #tensorboard --logdir="drive/mydrive/logs"
#
# # 要点：
# #
# # 1、每条线一个单独的文件夹
# #
# # 2、每条线一个单独的writer
# #
# # 3、表格名必须相同
import matplotlib.pyplot as plt
def draw_roc(y_true,y_score):
    # # kitsune绘制roc曲线的代码
    # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # roc_auc = auc(fpr, tpr)
    #
    # lowindex=find_nearest(fpr.tolist(),0.05)
    # print("低误报率:",thresholds[lowindex],fpr[lowindex])
    #
    # #以下代码打印tpr-fpr最大时的阈值
    # # lits.index(最大值) #返回这个最大值在list中的索引
    # maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
    # # print(maxindex)  # recall, FPR  约登指数0.05291631
    # print(thresholds[maxindex])  # decision_function生成的置信度来说



    # 绘图
    # 绘制ROC曲线
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)#lw代表线宽，label代表图例，注意x和y的输入要用np.array()
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') #linestyle
    plt.scatter(fpr[maxindex], tpr[maxindex], c="black", s=30)#打点
    plt.scatter(fpr[lowindex],tpr[lowindex],c="red",s=30)
    # for i in range(len(fpr.tolist())):
    #     plt.scatter(fpr[i],tpr[i],c="black",s=10)

    plt.xlim([0.0, 1.0])#x和y轴的作图范围
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")#将图例显示在右下角
    plt.show()