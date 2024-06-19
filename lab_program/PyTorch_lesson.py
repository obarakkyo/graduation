"""
PyTorchを使ってTLSHのCSVファイルを読み込んで
学習してみる。
"""

import torch
from torch import nn, optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


"""ニューラルネットワークの定義"""
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size):
        super(SimpleNN, self).__init__()
        self.fc1   = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Linear(hidden_size, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3   = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
    





"""訓練データとテストデータに分割する関数"""
def dowmload_and_split_dataset(csv_path: str) -> pd.DataFrame:
    ### CSVからデータフレームに変換 ###
    df = pd.read_csv(csv_path, index_col=0)
    print('-----csv.head-----', df.head())
    print('type(csv) = ', type(df))
    print('csv.shape = ', df.shape)

    ### 説明変数と目的変数に分割　###
    x = df.iloc[:, 0:-1].values
    y = df.loc[:, ["LABEL"]].values
    print("x.shape = ", x.shape)
    print("y.shape = ", y.shape)
    # print('---x.head()---', x[0:5])
    # print('---y.head()---', y[0:5])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y, shuffle=True)
    print("x_train.shape = ", x_train.shape)
    print("x_test.shape = ", x_test.shape)
    print("y_train.shape = ", y_train.shape)
    print("y_test.shape = ", y_test.shape)

    # numpy配列をテンソルに変換
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long).squeeze()
    y_test = torch.tensor(y_test, dtype=torch.long).squeeze()

    return x_train, x_test, y_train, y_test



"""メイン関数"""
def main() -> None:
    print('#####This program learns TLSH using PyTorch!#####')

    #訓練データとテストデータに分割
    # csv_path = "../CSV/anything/tlsh_csv_doc2vec_2label.csv"
    csv_path = "../CSV/dataset1CSV/ascii/tlsh_ascii_1split_2label.csv"
    # csv_path = "../CSV/dataset1CSV/ascii/tlsh_ascii_4split_2label.csv"
    # csv_path = "../CSV/dataset1CSV/doc2vec/tlsh_csv_doc2vec_4spilit_18dimention_2label.csv"

    x_train, x_test, y_train, y_test = dowmload_and_split_dataset(csv_path)


    #ハイパーパラメータの設定
    input_size   = 72
    hidden_size  = 36
    hidden_size2 = 72
    output_size = 2
    learning_rate = 0.001
    num_epochs = 200

    #モデルの初期化
    model = SimpleNN(input_size, hidden_size, hidden_size2, output_size)
    losses = []

    #損失関数とオプティマイザの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #トレーニング
    for epoch in range(num_epochs):
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        losses.append(loss.item())
        #逆伝搬とオプティマイザのステップ
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    print('Training finished.')

    #モデルの評価
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
        print(f'Accuracy on test data: {accuracy * 100:.2f}%')
    
    #損失関数プロット
    plt.plot(losses)
    plt.show()




if __name__ == "__main__":
    main()
