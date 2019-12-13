# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            
            #RuntimeError: size mismatch, m1: [8 x 12544], m2: [9216 x 4096] 
            # at /pytorch/aten/src/TH/generic/THTensorMath.cpp:197
        
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            #第一引数が入力チャネル　第二引数が出力チャネル数　縦かける横は自分で計算しないといけない
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
            
    def forward(self, x):
        y = self.features(x)
        return y

class Decoder(nn.Module):  #分類器　回帰でも使う
    def __init__(self, fc_size):
        super().__init__()
        #fc_sizeで計算した形状を指定
        self.decode_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(fc_size, 4096), 
            #Linearが恒等関数
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 6)  #nn.Linear(4096, 座標数)
            #ここが最後の出力層　ここのクラス分類を連続値にして90個出力

        )
    def forward(self, x):
        y = self.decode_layers(x)
        return y

class Classifier(nn.Module):  #分類器　回帰でも使う
    def __init__(self, fc_size):
        super().__init__()
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(fc_size, 4096)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(4096, 4)

    def forward(self, x):
        y = self.relu1(self.linear1(self.dropout1(x)))
        y = self.relu2(self.linear2(self.dropout2(y)))
        y = self.linear3(y)
        y=F.log_softmax(y)
        return y

class AlexNet(nn.Module):
    def __init__(self, hidden_size=1024):#
        super(AlexNet, self).__init__()
        
        self.features = Encoder()#
        size_check = torch.FloatTensor(512, 3, 256, 256) 
        #これだとbatchが10なんじゃないか？　torch.FloatTensor(8, 3, 256, 256)が正しい？ 
        #結果：RuntimeError: The size of tensor a (10) must match the size of tensor b (90) at non-singleton dimension 1

        #(8, 3, 224, 224)
        #結果：RuntimeError: size mismatch, m1: [10 x 12544], m2: [9216 x 4096] at /pytorch/aten/src/TH/generic/THTensorMath.cpp:197

        fc_size = self.features(size_check).view(size_check.size(0), -1).size()[1]#
        #view(size_check.size(0), -1) ここで入力の順番を逆転させてるから実際の入力も逆転させないといけない？
        self.linear=nn.Linear(fc_size, hidden_size * 9)
        self.Decoder = Decoder(hidden_size)
        self.classifier = Classifier(hidden_size)

        self.hidden_size=hidden_size

    def forward(self, x):
        x = self.features(x)
        # x: (batch_size, rgb, height, width)
        x = x.view(x.size(0), -1) # (bs, fc_size)
        x = self.linear(x) # (bs, 9 * hiddensize)
        x = x.view(x.size(0)*9,self.hidden_size)
        classes = self.classifier(x)
        positions = self.Decoder(x)
        return classes, positions

#########################################################################################

"""
# coding:utf-8
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            #第一引数が入力チャネル　第二引数が出力チャネル数　縦かける横は自分で計算しないといけない
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),


            #192  63出力にして空ノードはnullにするか？→-100

            #ここの最後の出力を変えればいいんじゃないか
            #出力させたい数に合わせて
            #10 9 90
        )

    def forward(self, x):
        y = self.features(x)
        return y

class Classifier(nn.Module):
    def __init__(self, fc_size):
        super().__init__()
        #fc_sizeで計算した形状を指定
        self.classifier = nn.Sequential(
        	#対象が複数ある場合に、並んでいる順番に処理することや、連続して立て続けに処理することを意味する。	
            nn.Dropout(p=0.5),
            nn.Linear(fc_size, 4096), 
            #Linearが恒等関数
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 90)  #nn.Linear(4096, num_classes)
            #ここが最後の出力層　ここのクラス分類を連続値にして90個出力

        )
    def forward(self, x):
        y = self.classifier(x)
        return y

class AlexNet(nn.Module):
    def __init__(self, num_classes, fc_size):
        super(AlexNet, self).__init__()
        
        self.features = Encoder()
        size_check = torch.FloatTensor(10, 3, 224, 224) 
        fc_size = features(size_check).view(size_check.size(0), -1).size()[1]
        self.classifier = Classifier(fc_size)

    def forward(self, x):
        x = self.features(x)
        # x: (batch_size, rgb, height, width)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
"""