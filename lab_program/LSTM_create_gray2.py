"""
DNN多段化。
より論文に近い形のNNクラスを用いた画像生成。
"""
import torch 
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

"""シグモイド関数"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


"""シーケンス作成関数"""
def make_sequence_data(y, num_sequence):
    num_data = len(y)
    seq_data = []
    target_data = []
    for i in range(num_data - num_sequence):
        seq_data.append(y[i:i+num_sequence])                    #訓練データ
        target_data.append(y[i+1:i+num_sequence+1])  #テストデータ
    seq_arr = np.array(seq_data)
    target_arr = np.array(target_data)
    return seq_arr, target_arr

"""データをシャッフルする関数"""
def shuffled_data_func(train_datalist, test_datalist):
    #データのインデックス
    indices = list(range(len(train_datalist)))
    #インデックスとデータをペアにする
    paired_data = list(zip(indices, train_datalist, test_datalist))
    #ペアごとシャッフル
    random.shuffle(paired_data)
    #分割
    shuffled_index, shuffled_train, shuffled_test = zip(*paired_data)
    return shuffled_index, shuffled_train, shuffled_test

"""メイン関数"""
def main():
    print("####################START####################")

    ###1.初期値の設定###
    # csv_path = "../CSV/dataset4CSV/ascii/2labelWithoutSummary.csv"
    csv_path = "../CSV/dataset4CSV/doc2vec/2labelWithoutSummary.csv"


    ###2.データの取得###
    df = pd.read_csv(csv_path, index_col=0)
    print("-----df.head-----")
    print("df.shape = ", df.shape)
    # print(df.head())
    print("-----------------\n")

    ###3.特徴量の抽出###
    features_data = df.iloc[:, 0:-1]
    print("features_data.shape = ", features_data.shape)


    ###4.RNN（多段化）の定義###
    class RNN(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.h = self.c = 0
            self.x1 = nn.Linear(input_size, 10)
            self.lstm = nn.LSTM(input_size=10, hidden_size=100, num_layers=2, dropout=0.5)
            self.y = nn.Linear(100, 10)
        def forward(self, x):
            x = self.x1(x)
            # print("Linear_output = ", x.shape)
            x, (self.h, self.c) = self.lstm(x.unsqueeze(0))
            # print("LSTM_output = ", x.shape)
            output = self.y(x.squeeze(0))
            # print("Output.shape = ", output.shape)
            return output ,self.h, self.c
        


    ###学習フェーズ(試し)###
    # for i in range(0, df.shape[0]):
    for i in range(1):
        ###RNNクラスの宣言###
        model = RNN(input_size=10)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        index_name = df.iloc[i].name

        ###訓練データの作成###
        vector_list = features_data.iloc[i, :].to_numpy()   #API100個
        sequence_length = 10
        sequence_x, sequence_target = make_sequence_data(vector_list, sequence_length)


        ###データをシャッフルする###
        shuffled_index, shuffled_sequence_x, shuffled_sequence_target = shuffled_data_func(sequence_x, sequence_target)
        # print(shuffled_index)

        ###FloatTensor型へ変換###
        shuffled_sequence_x = torch.FloatTensor(shuffled_sequence_x)
        shuffled_sequence_target = torch.FloatTensor(shuffled_sequence_target)
        sequence_x = torch.FloatTensor(sequence_x)
        sequence_target = torch.FloatTensor(sequence_target)
        # print(sequence_x[0])
        # print(sequence_target[0])
        # print(shuffled_sequence_x.size())
        # print(shuffled_sequence_target.size())

        ###学習フェーズ###
        losses = []
        for q in range(50):
            for j in range(shuffled_sequence_x.shape[0]):
                optimizer.zero_grad()
                output, h, c = model(shuffled_sequence_x[j])
                # print("output = ",output)
                # print("target = \n", shuffled_sequence_target[j])
                loss = criterion(output, shuffled_sequence_target[j])
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                # if j % 10 == 0:
                #     print("Epoch : {}, loss : {}".format(j, loss.item()))
        # print("h.shape = ", h.shape)
        # print("h.shape[1] = ", h[1, :])
        # print("c.shape[1] = ", c[1, :])

        ###損失関数のプロット###
        plt.plot(losses)
        plt.show()

        ###学習モデルからパラメータを抽出###
        with torch.no_grad(): 
            h_list = []
            c_list = []
            for i in range(sequence_x.shape[0]):
                _, h, c = model(sequence_x[i])
                h = h[1].squeeze(0).tolist()
                c = c[1].squeeze(0).tolist()
                h_list.append(h)
                c_list.append(c)
            # print(h_list)
            # print("type = ", type(h_list))
            # print("len(h_list) = ", len(h_list[0]))
        
        ###グレースケールの変換###
        sigmoid_arr = sigmoid(np.array(h_list))
        scaled_arr = sigmoid_arr * 255
        scaled_arr = scaled_arr.astype(np.uint8)
        # print(sigmoid_arr)

        ###PILを使って画像を保存###
        # print(scaled_arr.shape)
        image = Image.fromarray(scaled_arr)
        image.show()
        # image_path = "../custom_datasets/dataset_4_gray_pictures/ascii" + "/" + index_name + ".png"
        # image_path = "../custom_datasets/dataset_4_gray_pictures/doc2vec" + "/" + index_name + ".png"
        # print(image_path)
        # image.save(image_path)

            



    print("####################END####################")

if __name__ == "__main__":
    main()