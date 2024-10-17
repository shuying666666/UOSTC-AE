import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image
np.random.seed(0)
import torch.nn.functional as F
from PIL import Image



#残差块定义
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)#第一层有可能要改变通道和尺寸，第二层不会。如果改变了那么一定用了conv3
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

#残差模块定义，每个模块使用两个残差块，每个残差块包含两层神经网络
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致，在我们的设计中直接写成false
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

#逆转残差块定义
class re_Residual(nn.Module):#反卷积（转置卷积）参数设置参见网页https://blog.csdn.net/djfjkj52/article/details/117738905
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(re_Residual, self).__init__()
        if use_1x1conv:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride,
                                            output_padding=1)  # 第一层有可能要改变通道和尺寸，第二层不会。如果改变了那么一定用了conv3
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.conv3 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride,output_padding=1)
        else:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride,
                                            output_padding=0)  # 第一层有可能要改变通道和尺寸，第二层不会。如果改变了那么一定用了conv3
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return (Y + X)#反卷积的时候，最后不用relu，但开始前要有一层



#逆转残差模块定义，每个模块使用两个残差块，每个残差块包含两层神经网络
def re_resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(re_Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(re_Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        ### Convolutional section
        # self.encoder_cnn = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, stride=3, padding=1),#输入通道，输出通道，卷积核边长，步长，填充
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2,stride=2),#池化窗口，池化步长
        #     nn.Conv2d(16,8,3,stride=2,padding=1),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2,stride=1)
        # )
        # self.decoder_cnn=nn.Sequential(
        #     nn.ConvTranspose2d(8, 16, 3, stride=2),  # 输入通道，输出通道，卷积核边长，步长，填充
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16,8,5,stride=3,padding=1),  # 池化窗口，池化步长
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(8,1,2, stride=2,padding=1),
        #     nn.Tanh()
        # )

        # 图片进入后的第一层定义
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0),  # 只增加通道数量
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 残差块导入
        self.net.add_module("resnet_block2", resnet_block(32, 64, 2))  # 14   加通道，降维
        self.net.add_module("resnet_block3", resnet_block(64, 128, 2))  # 7

        self.net.add_module("fc", nn.Sequential(torch.nn.Flatten(start_dim=1), nn.Linear(128 * 7 * 7, 5)))  # 全连接层添加

        self.net.add_module("re_fc", nn.Sequential(nn.Linear(5, 128 * 7 * 7),
                                              torch.nn.Unflatten(dim=1, unflattened_size=(128, 7, 7))))  # 全连接层添加
        self.net.add_module("r_resnet_block3", re_resnet_block(128, 64, 2))  # 7
        self.net.add_module("r_resnet_block2", re_resnet_block(64, 32, 2))  # 14

        self.net.add_module("end", nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),  # 只增加通道数量
            nn.BatchNorm2d(1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ))




    def forward(self,x):
        x=self.net(x)
        return x

# class ae(object):
#     def __init__(self):
#         self.AE_net = Autoencoder()
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.AE_net.to(device)
#         self.AE_net.load_state_dict(torch.load('./AE/ckt.pt'))
#
#
#
#     def __call__(self, img):
#         # img2 = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
#         # return img2
#
#         img=transforms.ToTensor()(img)
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#把图片放到gpu上
#         img = img.to(device)
#
#         with torch.no_grad():
#             img=self.AE_net(img)
#         img=img.to("cpu")
#         return img




# if __name__ == "__main__":
#     img=Image.open('E:/22.jpg')
#     img2=ae()(img)
#     img.show()
#     img2.show()
