# coding:utf-8
from fastprogress import master_bar, progress_bar
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLossFunction(nn.Module):
    def __init__(self):
        super(CustomLossFunction, self).__init__()
    def forward(self, classes, positions, class_list, pos_list):
        nll_loss = F.nll_loss(classes, class_list)
        mse_loss = F.mse_loss(positions, pos_list)
        loss = nll_loss + mse_loss
        #print(loss)
        return loss

class MyTrainer:
    def __init__(self, model, dataloaders, optimizer, device):
        self.model = model

        self.device = device
        self.model = self.model.to(device)

        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.mse_loss = nn.MSELoss()
        self.nll_loss = nn.NLLLoss()

        # add custom loss func
        self.customLossFunc = CustomLossFunction()

        self.epoch=0

        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []        

    def run(self, epoch_num):
        self.master_bar = master_bar(range(epoch_num))
        for epoch in self.master_bar:
            self.epoch += 1
            self.train()
            self.test()
            self.save()
            self.draw_graph()
    
    def train(self):
        self.iter(train=True)
    
    def test(self):
        self.iter(train=False)

    def iter(self, train):
        if train:
            self.model.train()
            dataloader = self.dataloaders.train
        else:
            self.model.eval()
            dataloader = self.dataloaders.test#valid
        
        total_loss = 0.
        total_acc = 0.

        data_iter = progress_bar(dataloader, parent=self.master_bar)
        for i, batch in enumerate(data_iter):
            image_list = batch["image"].to(self.device)
            class_list = batch["class"].to(self.device)
            pos_list = batch["pos"].to(self.device)

            # forward
            classes, positions = self.model(image_list)

            # calc loss
            class_list= class_list.view(-1)
            # classes = torch.view(-1, classes)
            pos_list=pos_list.view(-1,6)
            class_loss = self.nll_loss(classes, class_list)
            position_loss = self.mse_loss(positions, pos_list)
            #loss = class_loss+position_loss
            '''
            loss=CustomLossFunction()
            loss.forward(classes, positions, class_list, pos_list)
            '''
            loss = self.customLossFunc(classes, positions, class_list, pos_list)


            # backward
            if train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad
            class_predicts=torch.argmax(classes,dim=1)
            total_loss += loss.item()
            acc = (class_predicts == class_list).sum().item() / len(class_list)
            total_acc += acc


        if train:
            self.train_loss_list.append(total_loss)
            self.train_acc_list.append(total_acc)
        else:
            self.val_loss_list.append(total_loss)
            self.val_acc_list.append(total_acc)

        train = "train" if train else "test" #"valid"
        print("[Info] epoch {}@{}: loss = {}, acc = {}".format(self.epoch, train, total_loss/ (i + 1), total_acc/(i + 1)))
    
    def save(self, out_dir="./output"):
        model_state_dict = self.model.state_dict()

        checkpoint = {
            "model": model_state_dict,
            "epoch": self.epoch,
        }

        model_name = "pose_acc_{acc:3.3f}.chkpt".format(
            acc = self.val_acc_list[-1]
        )
        torch.save(checkpoint, model_name)
    
    def draw_graph(self):
        x = np.arange(self.epoch)
        y = np.array([self.train_acc_list, self.val_acc_list]).T
        plots = plt.plot(x, y)
        plt.legend(plots, ("train", "test"), loc="best", framealpha=0.25, prop={"size": "small", "family": "monospace"})
        plt.xlabel("Epoch")
        #plt.tight_layer()
        plt.savefig("graph.png")
        # plt.show()
