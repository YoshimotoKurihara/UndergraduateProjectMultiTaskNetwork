# coding:utf-8
#export PYTHONPATH=/usr/local/lib/python3.6/dist-packages
#毎回これを実行
#インタープリタはpython3.7.4のbase condaのやつを使う
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
    IMAGE_PATH = "/home/gonken2019/Desktop/subProject/images"#
    LABELS_PATH = "/home/gonken2019/Desktop/subProject/labels/"#
    BATCH_SIZE = 512
    #BATCH_SIZE = 10
    #RuntimeError: size mismatch, m1: [10 x 12544], m2: [9216 x 4096] at /pytorch/aten/src/TH/generic/THTensorMath.cpp:197
    #9216×4096=37748736
    #37748736÷12544=3009.306122449
    #4096=2**12

    #BATCH_SIZE = 8
    #RuntimeError: size mismatch, m1: [8 x 12544], m2: [9216 x 4096] at /pytorch/aten/src/TH/generic/THTensorMath.cpp:197

    NUM_EPOCH = 50 #多くて20~30

    if torch.cuda.is_available():
        device = "cuda";
        print("[Info] Use CUDA")
    else:
        device = "cpu"
    model = AlexNet()
    dataloaders = Dataloaders(IMAGE_PATH, LABELS_PATH, BATCH_SIZE)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    #lossがnanになるのはよくあるので、こういうときはoptimizerを変えるか学習率変えるかするといい

    trainer = MyTrainer(model, dataloaders, optimizer, device)

    trainer.run(NUM_EPOCH)#

main()#

###########################################################################################

"""
# coding:utf-8
#export PYTHONPATH=/usr/local/lib/python2.7/dist-packages
#毎回これを実行
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
    IMAGE_PATH = ""
    LABELS_PATH = ""
    BATCH_SIZE = 8
    NUM_EPOCH = 25 #多くて20~30

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AlexNet()
    dataloaders = Dataloaders(IMAGE_PATH, LABELS_PATH, BATCH_SIZE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    trainer = MyTrainer(model, dataloaders, optimizer, criterion, device)

    trainer.run()

if __name__ is "__main__":
    main()

    """"""__name__はPythonプリンタでPythonスクリプトを読み込むと自動的に作成される。

        Pythonスクリプトを直接実行した時には、そのスクリプトファイルは「__main__」という名前のモジュールとして認識される
        そのため、スクリプトファイルを直接実行すると__name__変数の中に自動で'__main__'という値が代入される

        要するに
        「if __name__ == '__main__':」の意味は、「直接実行された場合のみ実行し、それ以外の場合は実行しない」"""