"""
データセット６用のシンプルなCNNを実装。
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import time


def main():
    """共通の初期化変数"""
    ngram_list = ["2gram", "3gram"]
    Boolean_list = [False, True]

    """ASCIIの時"""
    # vectorizer = "ascii"
    # for current_bool in Boolean_list:
    #     for current_ngram in ngram_list:
    #         print(f"[{vectorizer}] {current_bool}&{current_ngram}")
    #         if current_ngram == "2gram":
    #             SimpleCNN_Dataset6(current_vectorizer=vectorizer, current_ngram=current_ngram, 
    #                                current_bool=current_bool, ngram_dimention=129)
    #         elif current_ngram == "3gram":
    #              SimpleCNN_Dataset6(current_vectorizer=vectorizer, current_ngram=current_ngram, 
    #                                current_bool=current_bool, ngram_dimention=128)
    #         else:
    #             SimpleCNN_Dataset6(current_vectorizer=vectorizer, current_ngram=current_ngram, 
    #                                current_bool=current_bool, ngram_dimention=130)
    
    """BUCKETの時"""
    # vectorizer = "bucket"
    # bucket_len = [64, 128]
    # for current_bool in Boolean_list:
    #     for current_ngram in ngram_list:
    #         for current_len in bucket_len:
    #             print(f"[Bucket]vectorizer={vectorizer}, bool={current_bool}, ngram={current_ngram}, len={current_len}")
    #             if current_ngram == "2gram":
    #                 SimpleCNN_Dataset6(current_vectorizer=vectorizer, current_bool=current_bool,
    #                                current_ngram=current_ngram, current_bucketlen=current_len, ngram_dimention=129)
    #             elif current_ngram == "3gram":
    #                 SimpleCNN_Dataset6(current_vectorizer=vectorizer, current_bool=current_bool,
    #                                current_ngram=current_ngram, current_bucketlen=current_len, ngram_dimention=128)
    #             else:
    #                 SimpleCNN_Dataset6(current_vectorizer=vectorizer, current_bool=current_bool,
    #                                current_ngram=current_ngram, current_bucketlen=current_len, ngram_dimention=130)
                    
    """Doc2Vecの時"""
    # vectorizer = "doc2vec"
    # SimpleCNN_Dataset6(current_vectorizer=vectorizer)


    """TF-IDFの時"""
    vectorizer = "tfidf"
    for current_ngram in ngram_list:
        SimpleCNN_Dataset6(current_vectorizer=vectorizer, current_ngram=current_ngram, ngram_dimention=130)

            



# メイン関数 #
def SimpleCNN_Dataset6(current_vectorizer=None, current_ngram=None, current_bool=False, current_bucketlen=64, ngram_dimention=130):
    print("==========シンプルなCNNによる学習==========")

    """データセットの取得"""
    ## ASCII ##
    if current_vectorizer == "ascii":
        vectorizer_name = "ascii"
        n_gram = current_ngram
        LogBoolean = current_bool

        saving_file_path = f"../experiment/dataset6/SimpleCNN/Log{LogBoolean}/{vectorizer_name}/{n_gram}/report.txt"
        saving_plot_path   = f"../experiment/dataset6/SimpleCNN/Log{LogBoolean}/{vectorizer_name}/{n_gram}/importances.png"
        target_csv = f"../CSV/dataset6CSV/{vectorizer_name}/{n_gram}_Log{LogBoolean}_2label.csv"
        print(f"target_csv = {target_csv}")
    ## BucketPosition ##
    elif current_vectorizer == "bucket":
        vectorizer_name = "bucket"
        bucket_len =current_bucketlen
        n_gram = current_ngram
        LogBoolean = current_bool

        saving_file_path = f"../experiment/dataset6/SimpleCNN/Log{LogBoolean}/{vectorizer_name}/{bucket_len}/{n_gram}/report.txt"
        saving_plot_path   = f"../experiment/dataset6/SimpleCNN/Log{LogBoolean}/{vectorizer_name}/{bucket_len}/{n_gram}/importances.png"
        target_csv = f"../CSV/dataset6CSV/{vectorizer_name}/{n_gram}_PositionBucket_{bucket_len}_Log{LogBoolean}.csv"
    ## Doc2Vec ##
    elif current_vectorizer == "doc2vec":
        saving_file_path = f"../experiment/dataset6/SimpleCNN/doc2vec/report.txt"
        saving_plot_path = f"../experiment/dataset6/SimpleCNN/doc2vec/importances.png"
        target_csv = "../CSV/dataset6CSV/doc2vec/2label.csv"
    ## TF-IDF ##
    elif current_vectorizer == "tfidf":
        n_gram = current_ngram

        saving_file_path = f"../experiment/dataset6/SimpleCNN/tfidf/{n_gram}/report.txt"
        saving_plot_path = f"../experiment/dataset6/SimpleCNN/tfidf/{n_gram}/importances.png"
        target_csv = f"../CSV/dataset6CSV/tfidf/max100_{n_gram}_2label.csv"
    else:
        print("ベクトル化手法をしてしてください。")
        exit()


    """訓練データとテストデータの分割"""
    # CSVデータの読み込み
    df = pd.read_csv(target_csv, index_col=0)
    print(df.head(5))

    # 訓練データとテストデータの分割
    X = df.drop("LABEL", axis=1).values
    Y = df["LABEL"].values
    indices = df.index  # インデックス名を保持

    x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, Y, indices, test_size=0.2, random_state=123, stratify=Y, shuffle=True
    )

    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)

    



    """カスタムデータセットの定義"""
    class CustomDataset(Dataset):
        def __init__(self, features, labels, indices):
            self.features = torch.tensor(features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.indices = indices  # インデックス名を保持

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx], self.indices[idx]  # インデックス名も返す
    
    """データセットとデータローダーの作成"""
    train_dataset = CustomDataset(x_train.reshape(-1, 1, ngram_dimention, 1), y_train, train_indices)
    test_dataset = CustomDataset(x_test.reshape(-1, 1, ngram_dimention, 1), y_test, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True)



    """CNNモデルの定義部分"""
    class SimpleCNN(nn.Module):
        def __init__(self, cutom_dimention):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 1), stride=1, padding=(1, 0))  # 高さ方向の1D畳み込み
            self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=1, padding=(1, 0))
            self.fc1 = nn.Linear(64 * cutom_dimention * 1, 128)  # 出力をフラット化
            self.fc2 = nn.Linear(128, 50)
            self.fc3 = nn.Linear(50, 2)  # 2クラス分類用出力層

            # 初期化
            self._initialize_weights()
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # Conv2dレイヤーにはHe初期化（ReLUに適している）
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    # 全結合層にザビエル初期化（特定の活性化関数に依存しない）
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)  # 全結合層のためにフラット化
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    """学習用の変数群"""
    input_size = x_train.shape[1] #入力サイズ
    num_epochs = 100             #エポック数
    num_trials = 10             #総試行回数
    all_accuracies = []
    all_labels = []
    all_predictions = []
    all_incorrect_indices = [] # 誤分類インデックス名を保存
    incorrect_indices = []

    with open(saving_file_path, mode="w", encoding="utf-8") as f:
        start_time = time.time()
        for trial in range(num_trials):
            # モデルのインスタンス作成
            model = SimpleCNN(ngram_dimention)

            # 損失関数と最適化アルゴリズムの設定
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # 損失記録リスト
            train_losses = []
            test_losses = []

            # トレーニングループ
            for epoch in range(num_epochs):
                model.train()
                running_train_loss = 0.0
                for inputs, labels, _ in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_train_loss += loss.item()

                avg_train_loss = running_train_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                # テストデータの損失を計算
                model.eval()
                running_test_loss = 0.0
                with torch.no_grad():
                    for inputs, labels, _ in test_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        running_test_loss += loss.item()
                
                avg_test_loss = running_test_loss / len(test_loader)
                test_losses.append(avg_test_loss)
            
            # テストの実行
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                incorrect_indices = []
                for i, (inputs, labels, indices) in enumerate(test_loader):
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # 予測とラベルを記録
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
                    
                    # 誤分類のインデックス名を記録
                    incorrect_mask = (predicted != labels)
                    batch_incorrect_indices = [indices[j] for j in incorrect_mask.nonzero(as_tuple=True)[0].tolist()]
                    incorrect_indices.extend(batch_incorrect_indices)
            
            all_incorrect_indices.append(incorrect_indices)

            accuracy = 100 * correct / total
            all_accuracies.append(accuracy)
            print(f'Accuracy of the model on the test set: {accuracy}%', file=f)

        end_time = time.time()
        print(f"訓練時間 = {end_time - start_time}s", file=f)
        # 10回の平均を表示
        average_accuracy = sum(all_accuracies) / num_trials
        print(f'=====Average Accuracy over {num_trials} trials: {average_accuracy:.2f}%=====', file=f)

        # 評価レポートを表示
        print("\nClassification Report (averaged over trials):", file=f)
        print(classification_report(all_labels, all_predictions, target_names=['Class 0', 'Class 1']), file=f)

        # 混同行列を表示
        print("\nConfusion Matrix (accumulated over trials):")
        print(confusion_matrix(all_labels, all_predictions), file=f)

        # 誤分類のインデックス名を表示
        print("\nIncorrectly Classified Indices (accumulated over trials):", file=f)
        for i in range(len(all_incorrect_indices)):
            print(f"======{i+1}回目に間違えたもの 全部で{len(all_incorrect_indices[i])}個====", file=f)
            for j in range(len(all_incorrect_indices[i])):
                print(f" ・ {all_incorrect_indices[i][j]}", file=f)





if __name__ == "__main__":
    main()