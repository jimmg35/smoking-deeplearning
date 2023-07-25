import cv2
import numpy as np
from src.dataset import DigitDataset
from src.model import MLP
from torch.utils.data import DataLoader
import torch
from torch import nn

## 讀取資料
trainingSet = DigitDataset(r'./data/MNIST - JPG - training')
testingSet = DigitDataset(r'./data/MNIST - JPG - testing')

## 設定超參數
# 所有資料進入類神經網路一次，稱為一個epoch
EPOCH = 100
# 每次拿多少筆資料更新類神經網路
BATCH_SIZE = 1024
# 每個EPOCH更新參數的次數
STEP_PER_EPOCH = len(trainingSet) // BATCH_SIZE
# 學習率
LEARNING_RATE = 0.01


## 建立模型
model = MLP().to('cuda')

# 優化器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# 損失函數(交叉熵，專門用於分類任務)
loss_function = nn.CrossEntropyLoss()


# 將training set送入至data loader
train_dataloader = DataLoader(
    trainingSet, batch_size=BATCH_SIZE, shuffle=True
)
# 將testing set送入data loader
test_dataloader = DataLoader(
    testingSet, batch_size=BATCH_SIZE, shuffle=True
)

# 開始訓練
for epoch in range(EPOCH):

    train_loss = 0.0
    train_step = 0
    val_loss = 0.0
    val_step = 0

    for data, target in train_dataloader:
        # 將資料讀入至cpu或gpu
        data, target = data.to('cuda'), target.to('cuda')
        # 清除梯度
        optimizer.zero_grad()
        # 進行預測
        pred = model(data)
        # 計算殘差(loss)
        loss = loss_function(pred, target.double())
        # 反向傳播(更新模型參數)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_step += 1
        print(f"epoch : {epoch+1} | step : {train_step} | loss : {loss.item()}")

    for data, target in test_dataloader:
        # 將資料讀入至cpu或gpu
        data, target = data.to('cuda'), target.to('cuda')
        pred = model(data)
        loss = loss_function(pred, target)
        val_loss += loss.item()
        val_step += 1

        print(f"-epoch : {epoch+1} | step : {val_step} | loss : {loss.item()}")
    
    mean_train_loss = train_loss / train_step
    mean_val_loss = val_loss / val_step

    print(f"=epoch : {epoch+1} | training loss : {mean_train_loss} | validation loss : {mean_val_loss}")

    




