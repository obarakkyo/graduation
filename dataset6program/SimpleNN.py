"""
シンプルなNNを用いて、学習してみる.
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def main():
    # 対象CSVの取得 #
    ### ASCII ###
    # target_csv_path = "../CSV/dataset6CSV/ascii/1gram_LogTrue_2label.csv"

    ### Bucket配列系 ###
    # target_csv_path = "../CSV/dataset6CSV/bucket/Position_128_2label.csv"
    # target_csv_path = "../CSV/dataset6CSV/bucket/2gram_PositionBucket_128.csv"
    # target_csv_path = "../CSV/dataset6CSV/bucket/3gram_PositionBucket_128.csv"

    # target_csv_path = "../CSV/dataset6CSV/bucket/Position_64_2label.csv"
    target_csv_path = "../CSV/dataset6CSV/bucket/1gram_PositionBucket_64_LogTrue.csv"

    # target_csv_path = "../CSV/dataset6CSV/bucket/1gram_PositionBucket_128_LogTrue.csv"



    ### Doc2Vec ###
    # target_csv_path = "../CSV/dataset6CSV/doc2vec/2label.csv"


    ### TF-IDF ###
    # target_csv_path = "../CSV/dataset6CSV/tfidf/max100_3gram_2label.csv"



    df = pd.read_csv(target_csv_path, index_col=0)
    print(df.head(5))

    # 訓練データとテストデータの分割 #
    X = df.drop("LABEL", axis=1).values
    Y = df["LABEL"].values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123, stratify=Y, shuffle=True)
    print(f"x_train = {x_train.shape}")
    print(f"x_test  = {x_test.shape}")
    print(f"y_train = {y_train.shape}")
    print(f"y_test  = {y_test.shape}")

    # カスタムデータセットの定義 #
    class CustomDataset(Dataset):
        def __init__(self, features, labels):
            self.features = torch.tensor(features, dtype=torch.float32)
            self.labels   = torch.tensor(labels, dtype=torch.long)
        
        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    
    # データセットとデータローダーの作成 #
    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=50, shuffle=True)

    # NNの定義 #
    class SimpleNN(nn.Module):
        def __init__(self, input_size):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 200)
            self.fc2 = nn.Linear(200, 50)
            self.fc3 = nn.Linear(50, 2) 
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
    # モデルのインスタンス作成 #
    input_size = x_train.shape[1]
    num_epochs = 100
    num_trials = 10
    all_accuracies = []
    all_labels = []
    all_predictions = []
    all_incorrect_indices = []

    for trial in range(num_trials):
        # モデルのインスタンス作成 #
        model = SimpleNN(input_size)

        # 損失関数と最適化アルゴリズムの設定 #
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)


        # 損失記録リスト #
        train_losses = []
        test_losses  = []

        # トレーニングループ #
        for epoch in range(num_epochs):
            model.train()
            running_train_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()

            # エポックごとの平均訓練損失を計算して記録
            avg_train_loss = running_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # テストデータの損失を計算
            model.eval()
            running_test_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_test_loss += loss.item()
            
            # エポックごとの平均テスト損失を記録
            avg_test_loss = running_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)

            # if epoch%10 == 0:
            #     print(f"Epoch [{epoch+1}/{num_epochs}], Loss : {loss.item():.4f}")
        
            # 損失の推移をプロット
            # plt.figure(figsize=(10, 5))
            # plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
            # plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
            # plt.xlabel("Epoch")
            # plt.ylabel("Loss")
            # plt.title("Training and Testing Loss")
            # plt.legend()
            # plt.show()

        
        # テストの実行 #
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                 # 予測とラベルを記録
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                # 誤分類のインデックスを記録
                batch_incorrect_indices = (predicted != labels).nonzero(as_tuple=True)[0].cpu().numpy()
                all_incorrect_indices.extend((i * test_loader.batch_size + idx) for idx in batch_incorrect_indices)

        accuracy = 100 * correct / total
        all_accuracies.append(accuracy)
        print(f'Accuracy of the model on the test set: {accuracy}%')
    
    # 10回の平均を表示 #
    average_accuracy = sum(all_accuracies) / num_trials
    print(f'=====Average Accuracy over {num_trials} trials: {average_accuracy:.2f}%=====')

    # 評価レポートを表示
    print("\nClassification Report (averaged over trials):")
    print(classification_report(all_labels, all_predictions, target_names=['Class 0', 'Class 1']))

    # 混同行列を表示
    print("\nConfusion Matrix (accumulated over trials):")
    print(confusion_matrix(all_labels, all_predictions))

    # 誤分類のインデックスを表示
    print("\nIncorrectly Classified Indices (accumulated over trials):")
    print(all_incorrect_indices)





if __name__ == "__main__":
    main()