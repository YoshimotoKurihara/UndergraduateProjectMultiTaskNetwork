# coding:utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os

#同ディレクトリ内の別のファイルからのクラスの読み込み
from model import AlexNet
from dataloader import Dataloaders
from trainer import MyTrainer

def main():
    IMAGE_PATH = "/home/gonken2019/Desktop/subProject/images"
    LABELS_PATH = "/home/gonken2019/Desktop/subProject/labels/"
    BATCH_SIZE = 512
    
    NUM_EPOCH = 50 #多くて20~30

    if torch.cuda.is_available():
        device = "cuda";
        print("[Info] Use CUDA")
    else:
        device = "cpu"
    model = AlexNet()
    dataloaders = Dataloaders(IMAGE_PATH, LABELS_PATH, BATCH_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    trainer = MyTrainer(model, dataloaders, optimizer, device)
    trainer.run(NUM_EPOCH)

if __name__ is "__main__":
    main()
