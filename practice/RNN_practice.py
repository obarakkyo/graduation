"""
RNNの練習用
https://ex-ture.com/blog/2021/01/12/pytorch-rnn/

np.random.normal(平均, 標準偏差, 配列のサイズ)
"""
"""必要なライブラリ"""
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt






def main():
    """GPUチェック"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device = {}".format(device))

    """訓練データ"""
    x = np.linspace(0, 4*np.pi) #0~4πまでの等間隔な点を用意
    sin_x = np.sin(x) + np.random.normal(0, 0.3, len(x)) #sin関数に乱数でノイズを加える
    plt.plot(x, sin_x)
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.show()

    """ハイパーパラメーター"""
    n_time = 10                 #10個の時系列データを１セット
    n_sample = len(x) - n_time  #len(x)=50, n_sample=50-10=40

    """データを格納する空の配列を準備"""
    input_data = np.zeros((n_sample, n_time, 1)) #入力値を格納する[40, 10, 1]の配列
    correct_data = np.zeros((n_sample, 1))       #正解値を格納する[40, 1]の配列

    """前処理"""
    for i in range(n_sample):
        input_data[i] = sin_x[i:i+n_time].reshape(-1, 1) #10個の時系列データを１列にして配列へ格納
        correct_data[i] = [sin_x[i+n_time]]              #入力値の１つ後を正解値とした
    input_data = torch.FloatTensor(input_data)
    correct_data = torch.FloatTensor(correct_data)

    """バッチデータの準備"""
    dataset = TensorDataset(input_data, correct_data)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True) #１回の処理で８組のデータを扱う


    """モデルの定義"""
    class RNN(nn.Module):
        def __init__(self):
            super(RNN, self).__init__()
            self.rnn = nn.RNN(1, 64, batch_first=True)
            self.fc = nn.Linear(64, 1)
        def forward(self, x):
            batch_size = x.size(0)
            x = x.to(device)
            x_rnn, hidden = self.rnn(x,None)
            x = self.fc(x_rnn[:, -1, :])
            return x
    model = RNN().to(device)

    """最適化手法の定義"""
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    """誤差を記録する空の配列を用意"""
    record_loss_train = []


    """学習"""
    for i in range(401):
        model.train()
        loss_train = 0
        for j, (x, t) in enumerate(train_loader):
            loss = criterion(model(x), t.to(device))
            loss_train += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_train /= j+1
        record_loss_train.append(loss_train)
        if i%50 == 0:
            print("Epoch:", i, "Loss_Train:", loss_train)
            predicted = list(input_data[0].reshape(-1))
            model.eval()
            with torch.no_grad():
                for i in range(n_sample):
                    x = torch.tensor(predicted[-n_time:])
                    x = x.reshape(1, n_time, 1)
                    predicted.append(model(x)[0].item())
            plt.plot(range(len(sin_x)), sin_x, label="before")
            plt.plot(range(len(predicted)), predicted, label="predicred")
            plt.xlabel("x")
            plt.ylabel("sin(x)")
            plt.legend()
            plt.show()






if __name__ == "__main__":
    print("-----RNNの練習用のプログラム-----")
    main()