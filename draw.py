#此文件用于画图，绘制损失变化、top1准确、top5准确、学习率变化图
import torch
import matplotlib.pyplot as plt #用于显示图片




path1024 = './result_analysis/t1/Mar26_22-11-49_amax/sess058141518all_1024.pth.tar'  # 模型读取路径
path512 = './result_analysis/t1/Mar26_23-19-01_amax/sess058141518all_0512.pth.tar'
path256='./result_analysis/t1/256/sess058141518all_0256.pth.tar'
result1024 = torch.load(path1024, map_location=torch.device('cpu'))
result512=torch.load(path512,map_location=torch.device('cpu'))
result256=torch.load(path256,map_location=torch.device('cpu'))


# 绘图
l1=result1024['loss_clr']
l2=result512['loss_clr']
l3=result256['loss_clr']
plt.figure()
lw = 2
# plt.plot(loss_clr, color='darkorange',
#          lw=lw)#lw代表线宽
plt.plot(l1, color='red',
         lw=lw,label='batch_size=1024')#lw代表线宽
plt.plot(l2, color='green',
         lw=lw,label='batch_size=512')#lw代表线宽
plt.plot(l3, color='blue',
         lw=lw,label='batch_size=256')#lw代表线宽

plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('The variation of loss function with training rounds')
plt.legend(loc="upper right")
plt.show()




# 绘图
l1=result1024['accuracy']
l2=result512['accuracy']
l3=result256['accuracy']
plt.figure()
lw = 2
# plt.plot(loss_clr, color='darkorange',
#          lw=lw)#lw代表线宽
plt.plot(l1, color='red',
         lw=lw,label='batch_size=1024')#lw代表线宽
plt.plot(l2, color='green',
         lw=lw,label='batch_size=512')#lw代表线宽
plt.plot(l3, color='blue',
         lw=lw,label='batch_size=256')#lw代表线宽

plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('The variation of accuracy with training rounds')
plt.legend(loc="lower right")
plt.show()




# 绘图
l1=result1024['accuracy_top5']
l2=result512['accuracy_top5']
l3=result256['accuracy_top5']
plt.figure()
lw = 2
# plt.plot(loss_clr, color='darkorange',
#          lw=lw)#lw代表线宽
plt.plot(l1, color='red',
         lw=lw,label='batch_size=1024')#lw代表线宽
plt.plot(l2, color='green',
         lw=lw,label='batch_size=512')#lw代表线宽
plt.plot(l3, color='blue',
         lw=lw,label='batch_size=256')#lw代表线宽

plt.xlabel('epoch')
plt.ylabel('accuracy_top5')
plt.title('The variation of accuracy_top5 with training rounds')
plt.legend(loc="lower right")
plt.show()


# 绘图
l1=result1024['lr']
l2=result512['lr']
l3=result256['lr']
plt.figure()
lw = 2
# for i
# plt.plot(loss_clr, color='darkorange',
#          lw=lw)#lw代表线宽
plt.plot(l1, color='red',
         lw=3,label='batch_size=1024',linestyle='--')#lw代表线宽
plt.plot(l2, color='green',
         lw=2,label='batch_size=512')#lw代表线宽
plt.plot(l3, color='blue',
         lw=lw,label='batch_size=256')#lw代表线宽

plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.title('The variation of learning rate with training rounds')
plt.legend(loc="lower right")
plt.show()
















# # 绘制ROC曲线
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)#lw代表线宽
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.scatter(fpr[maxindex], tpr[maxindex], c="black", s=30)
# plt.scatter(fpr[lowindex],tpr[lowindex],c="red",s=30)
# # for i in range(len(fpr.tolist())):
# #     plt.scatter(fpr[i],tpr[i],c="black",s=10)
#
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
