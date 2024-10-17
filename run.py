import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import models
from torchvision.transforms import transforms
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
#from torchinfo import summary


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='mnist',
                    help='dataset name', choices=['stl10', 'cifar10','mnist'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',#训练轮次，一般50就够了
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1024, type=int,#训练batch
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,#学习率
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=200, type=int,#日志记录频率，不用tensorboard这个也用不到了
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


def main(checkpoint=False):#checkpoint用于断点续训
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    # 加载MNIST数据集
    train_data = torchvision.datasets.MNIST(root="./datasets",  # 数据路径
                                            train=True,  # 只使用训练数据集
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                            ]),
                                            # 把PIL.Image或者numpy.array数据类型转变为torch.FloatTensor类型                                                                                                        # 尺寸为Channel * Height * Width，数值范围缩小为[0.0, 1.0]
                                            download=False,  # 若本身没有下载相应的数据集，则选择True
                                            )
    train_loader = torch.utils.data.DataLoader(dataset=train_data,  # 传入的数据集
                              batch_size=args.batch_size,  # 每个Batch中含有的样本数量
                              shuffle=True,  # 是否对数据集重新排序
                              num_workers=args.workers, pin_memory=True, drop_last=True
                              )

    # 做训练集类别切分 https://www.saoniuhuo.com/question/detail-2685956.html
    idx = (train_data.targets != 0) & (train_data.targets != 5) & (train_data.targets != 8) & (
            train_data.targets != 14) & (train_data.targets != 15) & (train_data.targets != 18)
    train_data.targets = train_data.targets[idx]
    train_data.data = train_data.data[idx]

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)#基础resnet18加上投影头，首层通道数改变也在这里完成
    optimizer = torch.optim.Adam(
           model.parameters(), lr= args.lr,
         weight_decay=args.weight_decay)

    # 以下是断点读取程序
    if checkpoint:
        path = './checkpoint/checkpoint.pth.tar'  # 断点读取路径
        checkpoint = torch.load(path, map_location=args.device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # 优化器导入参数导致数据出现在cpu和gpu上，所以要把优化器数据转入gpu
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        print("history epoch:")
        print(checkpoint['epoch'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)#学习率调整作用,设置学习率下降策略

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)#simclr封装了具体的训练方案，没有更改模型结构。
        simclr.train(train_loader)
        # #注意，args.batch要改，train_loader也要换，两者缺一不可
        # args.batch_size=512
        # simclr.train(train_loader512)
        # args.batch_size=256
        # simclr.train(train_loader256)
        # print(summary(model,input_size=(1024,1,28,28)))
        # print(model)





if __name__ == "__main__":
    main()
