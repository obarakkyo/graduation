"""
目的：LSTMを用いてグレースケール画像を出力するプログラム

CSVファイルの読み込み
↓
LSTMを用いて学習
↓
隠れ層のパラメータを取得
↓
グレースケール画像に変更
"""

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# from sklearn.model_selection import train_test_split

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
    print("----------START----------\n")
    """初期値の設定"""
    csv_path = "../CSV/dataset4CSV/ascii/2labelWithoutSummary.csv"

    """データの取得"""
    df = pd.read_csv(csv_path, index_col=0)
    print("-----df.head-----")
    print("df.shape = ", df.shape)
    print(df.head())
    print("-----------------\n")
    

    # """訓練データとテストデータに分割"""
    # features_data = df.iloc[:, 0:-1]
    # label = df.loc[:, ["LABEL"]]
    # x_train, x_test, y_train, y_test = train_test_split(features_data, label, test_size=0.2, random_state=123, shuffle=True, stratify=label)
    # print("----訓練データとテストデータ----")
    # print("x_train.shape = {}".format(x_train.shape))
    # print("x_test.shape  = {}".format(x_test.shape))
    # print("y_train.shape = {}".format(y_train.shape))
    # print("y_test.shape  = {}".format(y_test.shape))
    # print("-------------------------------\n")

    """特徴量を抽出してくる"""
    features_data = df.iloc[:, 0:-1]

    """LSTMクラスの定義"""
    class LSTM(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.lstm  = nn.LSTM(1, hidden_size=self.hidden_size, num_layers=2, dropout=1)
            self_linear_first = nn.Linear(100, 50)
            self.linear_last = nn.Linear(hidden_size, 1)
        def forward(self, x):
            x, (h_n, c_n) = self.lstm(x) #シーケンス長、バッチサイズ、input_size
            x_last = x[-1]
            x = self.linear_last(x_last)
            return x, h_n, c_n
    
    """とりあえず３回やってみる"""
    for i in range(0, len(df), 240):
    # for i in range(1):
        index_name = features_data.iloc[i].name             #ファイル名
        vector_list = features_data.iloc[i, :].to_numpy()   #API100個
        # print(vector_list_t, type(vector_list_t), vector_list_t.size())

        """訓練データ作成"""
        sequence_length = 10
        sequence_x, sequence_target = make_sequence_data(vector_list, sequence_length)
        # print(sequence_x)
        # print(sequence_target)

        """FloatTensor型へ変換"""
        sequence_x = torch.FloatTensor(sequence_x)
        sequence_target = torch.FloatTensor(sequence_target)
        print(sequence_x.size())
        print(sequence_target.size())

        """入力次元の入れ替え"""
        sequence_x = sequence_x.permute(1, 0)               #torch.Size([90, 10])
        sequence_target = sequence_target.permute(1, 0)     #torch.Size([90, 1])

        """次元数（1）の付与"""
        sequence_x = sequence_x.unsqueeze(dim=-1)
        sequence_target = sequence_target.unsqueeze(dim=-1)
        print(sequence_x.size())            #torch.Size([10, 90, 1])
        print(sequence_target.size())       #torch.Size([1, 90, 1])


        """モデルの宣言"""
        model = LSTM(hidden_size=100)

        """損失関数と最適化アルゴリズム"""
        criterion = nn.MSELoss()
        # criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        """学習"""
        num_epoch = 100
        losses = []
        for epoch in range(num_epoch):
            optimizer.zero_grad()
            output, h_n, c_n = model(sequence_x)
            loss = criterion(output, sequence_target)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            print(epoch)
            if epoch % 10 == 0:
                print("Epoch : {}, loss : {}".format(epoch, loss.item()))
        
        """損失関数の描画"""
        # plt.plot(losses)
        # plt.show()

        """隠れ層の表示"""
        print("h_n = ", h_n.size()) #torch.Size([1, 450, 100])
        print()
        print("c_n = ", c_n.size()) #torch.Size([1, 450, 100])

        h_n = h_n[1].squeeze()
        print(h_n.size())
        h_n_np = h_n.detach().numpy()
        print(h_n_np.shape)

        """グレースケール画像にして表示してみる"""
        sigmoid_arr = sigmoid(h_n_np)
        print(sigmoid_arr)
        scaled_arr = sigmoid_arr * 255
        scaled_arr = scaled_arr.astype(np.uint8)

        # matplotlibを使って画像を表示
        # plt.imshow(scaled_arr, cmap='gray')
        # plt.colorbar()
        # plt.show()

         # PILを使って画像を保存
        print(scaled_arr.shape)
        image = Image.fromarray(scaled_arr)
        image_path = "../custom_datasets/dataset_4_gray_pictures" + "/" + index_name + ".png"
        print(image_path)
        image.save(image_path)







if __name__ == "__main__":
    main()