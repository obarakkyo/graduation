"""
LSTMの練習
https://qiita.com/sloth-hobby/items/93982c79a70b452b2e0a
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("ggplot")

from PIL import Image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def make_sequence_data(y, num_sequence):
    num_data = len(y)
    seq_data = []
    target_data = []
    for i in range(num_data - num_sequence):
        seq_data.append(y[i:i+num_sequence])                    #訓練データ
        target_data.append(y[i+num_sequence:i+num_sequence+1])  #テストデータ
    seq_arr = np.array(seq_data)
    target_arr = np.array(target_data)
    return seq_arr, target_arr


def main():
    """sin波の作成"""
    x = np.linspace(0, 499, 500) #始まり,終わり,分割数
    y = np.sin(x * 2 * np.pi / 50)
    # plt.plot(x, y)
    # plt.show()

    """訓練データ作成"""
    sequence_length = 40 #シーケンス長
    y_seq, y_target = make_sequence_data(y, sequence_length)
    print("y_seq.shape = ", y_seq.shape)        #(460, 40)
    print("y_target.shape = ", y_target.shape)  #(460, 1)

    """テストデータ作成"""
    num_test =10
    y_seq_train = y_seq[:-num_test]
    y_seq_test  = y_seq[-num_test:]
    y_target_train = y_target[:-num_test]
    y_target_test = y_target[-num_test:]
    print("y_seq_train.shape = ", y_seq_train.shape)        #(450, 40)
    print("y_seq_test.shape = ", y_seq_test.shape)          #(10, 40)
    print("y_target_train.shape = ", y_target_train.shape)  #(450, 1)
    print("y_target_train.shape = ", y_target_test.shape)   #(10, 1)

    """FloatTensor型の変換"""
    y_seq_t    = torch.FloatTensor(y_seq_train)
    y_target_t = torch.FloatTensor(y_target_train)

    """LSTMクラスの定義"""
    class LSTM(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size)
            self.linear = nn.Linear(hidden_size, 1)
        def forward(self, x):
            x, (h_n, c_n) = self.lstm(x) #シーケンス長、バッチサイズ、input_size
            x_last = x[-1]
            x = self.linear(x_last)
            return x, h_n, c_n
    
    """モデルの宣言"""
    model = LSTM(hidden_size=100)

    """損失関数と最適化アルゴリズム"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    """入力次元の入れ替え"""
    y_seq_t = y_seq_t.permute(1, 0)         #(40, 450)
    y_target_t = y_target_t.permute(1, 0)   #(40, 450)

    """次元数(1)の付与"""
    y_seq_t = y_seq_t.unsqueeze(dim=-1)         #torch.Size([40, 450, 1])
    y_target_t = y_target_t.unsqueeze(dim=-1)   #torch.Size([1, 450, 1])


    """学習"""
    num_epochs = 20
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output, h_n, c_n = model(y_seq_t)
        loss = criterion(output, y_target_t)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        if epoch % 10 == 0:
            print("Epoch : {}, loss : {}".format(epoch, loss.item()))

    """損失関数の描画"""
    # plt.plot(losses)
    # plt.show()

    """隠れ層の表示"""
    print("h_n = ", h_n.size()) #torch.Size([1, 450, 100])
    print()
    print("c_n = ", c_n.size()) #torch.Size([1, 450, 100])

    h_n = h_n.squeeze()
    print(h_n.size())
    h_n_np = h_n.detach().numpy()
    print(h_n_np.shape)

    """グレースケール画像にして表示してみる"""
    sigmoid_arr = sigmoid(h_n_np)
    print(sigmoid_arr)
    scaled_arr = sigmoid_arr * 255
    scaled_arr = scaled_arr.astype(np.uint8)

    # matplotlibを使って画像を表示
    plt.imshow(scaled_arr, cmap='gray')
    plt.colorbar()
    plt.show()

    # PILを使って画像を保存
    # image = Image.fromarray(scaled_arr)
    # image.save('sigmoid_grayscale_image.png')
    

    """テストデータで確認"""
    # y_seq_test_t = torch.FloatTensor(y_seq_test)
    # y_seq_test_t = y_seq_test_t.permute(1, 0)
    # y_seq_test_t = y_seq_test_t.unsqueeze(-1)   #torch.Size([40, 10, 1])

    # y_pred = model(y_seq_test_t)
    # plt.plot(x, y)
    # plt.plot(np.arange(490, 500), y_pred.detach())
    # plt.xlim(450.0, 500.0)
    # plt.show()


if __name__ == "__main__":
    print("LSTMの練習用プログラム")
    main()