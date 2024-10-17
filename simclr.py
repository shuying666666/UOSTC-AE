import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer=None
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)


    def info_nce_loss(self, features):#对比学习损失函数

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        scaler = GradScaler(enabled=self.args.fp16_precision)#用于混合精度梯度缩放，加速用的

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        loss_clr_pt=[]
        accuracy_pt=[]
        accuracy_top5_pt=[]
        lr_pt=[]



        for epoch_counter in range(self.args.epochs):
            # total_loss=0
            # total_loss_ae=0
            total_loss_clr=0
            total_accuracy=0
            total_accuracy_top5=0

            for images, _ in tqdm(train_loader):                    #tqdm代表进度条
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features,loss_ae = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss_clr = self.criterion(logits, labels)
                # loss=loss_clr+loss_ae

                self.optimizer.zero_grad()

                scaler.scale(loss_clr).backward()

                scaler.step(self.optimizer)
                scaler.update()
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                # total_loss += loss
                # total_loss_ae += loss_ae
                total_loss_clr += loss_clr
                total_accuracy+=top1[0]
                total_accuracy_top5+=top5[0]

            # #训练中间结果保存
            # if epoch_counter % 30 == 0:
            #     # save model checkpoints
            #     checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
            #     save_checkpoint({
            #         'epoch': self.args.epochs,
            #         'arch': self.args.arch,
            #         'state_dict': self.model.state_dict(),
            #         'optimizer': self.optimizer.state_dict(),
            #     }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            #日志写入log文件
            # total_loss=total_loss/len(train_loader)
            # total_loss_ae=total_loss_ae/len(train_loader)
            total_loss_clr=total_loss_clr/len(train_loader)
            total_accuracy=total_accuracy/len(train_loader)
            total_accuracy_top5=total_accuracy_top5/len(train_loader)
            logging.debug(f"Epoch: {epoch_counter}\tLoss_clr:{total_loss_clr.item()}\tTop1 accuracy: {total_accuracy.item()}\tTop5 accuracy: {total_accuracy_top5.item()}\t")#Loss: {total_loss}\tLoss_ae:{total_loss_ae}\t
            #日志写入pt列表
            loss_clr_pt.append(total_loss_clr.item())
            accuracy_pt.append(total_accuracy.item())
            accuracy_top5_pt.append(total_accuracy_top5.item())
            lr_pt.append(self.scheduler.get_lr()[0])










        #训练循环结束
        logging.info("Training has finished.")
        # 保存最终模型
        checkpoint_name = 'sess058141518all_{:04d}.pth.tar'.format(train_loader.batch_size)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_clr':loss_clr_pt,
            'accuracy':accuracy_pt,
            'accuracy_top5':accuracy_top5_pt,
            'lr':lr_pt
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

        self.writer.close()# 在训练结束后，确保所有的数据都被写入磁盘并关闭 writer











