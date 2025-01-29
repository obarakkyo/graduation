"""
RandomDataProject用のRandomForestプログラム．

ベースディレクトリ = graduation
"""
import time
import glob
import pandas as pd
import matplotlib.pyplot as plt

#ランダムフォレスト、グリッドリサーチ
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#混同行列、適合率、再現率、F値
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def RandomForestMain(status_dict=None):
    """
    メイン関数
    """
    # CSVの取得 #
    origin_df = pd.read_csv(status_dict["targetCSV"], index_col=0)
    print(f"\n##########targetCSV = {status_dict['targetCSV']}##########")
    print(f"origin_df.shape={origin_df.shape}")

    # 特徴量と正解ラベルの分割 #
    X = origin_df.iloc[:, :-1]
    y = origin_df.loc[:, ['LABEL']]
    print(f"X.shape = {X.shape}")
    print(f"y.shape = {y.shape}")

    # 学習データの分割 #
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y, shuffle=True)
    print("##訓練データとテストデータの形状##")
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)

    # グリッドサーチ用のハイパーパラメータ#
    parameters = {
        'n_estimators' : [i for i in range(50, 100, 5)],
        'max_features'  : ('sqrt', 'log2', None),
        'max_depth'   : [i for i in range(20, 50, 5)],
    }

    # モデルのインスタンス#
    model = RandomForestClassifier(class_weight="balanced", random_state=123)

    # グリッドサーチのインスタンス生成 #
    gridsearch = GridSearchCV(
        estimator = model,
        param_grid = parameters,
        scoring = 'accuracy',
        cv=5,
        n_jobs=-1,
        verbose=0,
    )

    # グリッドサーチ開始 #
    print("グリッドサーチスタート ")
    grid_start_time = time.time()
    gridsearch.fit(x_train, y_train.values.ravel())
    grid_end_time = time.time()

    # 訓練データを保存 #
    with open(status_dict["SavingPath"], mode="w", encoding="utf-8") as f:
        print("##########グリッドサーチ終了##########", file=f)

        # 処理時間を保存 #
        print(f"Time : {grid_end_time - grid_start_time}", file=f)

        #最適なパラメータの取得
        print('Best params : {}'.format(gridsearch.best_params_), file=f)
        print('Best Score  : {}'.format(gridsearch.best_score_), file=f)

        #最高性能のモデルを取得し、テストデータを分類
        best_model = gridsearch.best_estimator_
        y_pred = best_model.predict(x_test)
        print('best_model\'s score = ', best_model.score(x_test, y_test), file=f)

        #混同行列を表示
        print(confusion_matrix(y_test, y_pred), file=f)
        print(classification_report(y_test, y_pred), file=f)

        # 分類を間違えたインデックスの抽出
        misclassified_indices = y_test.index[y_test['LABEL'] != y_pred]

        # 分類を間違えたファイル名の表示
        print("Misclassified file names:", file=f)
        print(misclassified_indices, file=f)

    #重要な特徴量を可視化
    labels = x_train.columns
    importances = best_model.feature_importances_

    plt.figure(figsize = (10,6))
    plt.barh(y = range(len(importances)), width = importances)
    plt.yticks(ticks = range(len(labels)), labels = labels)
    plt.savefig(status_dict["SavingPlotPath"])
    # plt.show()




if __name__ == "__main__":
    # ASCII用 #
    # CsvPath = "CSV/RandomDatasetCSV/ascii/*.csv"
    # csvpaths = glob.glob(CsvPath)

    # for path in csvpaths[:]:
    #     # 保存用のファイル名作成 #
    #     filename = path.replace("CSV/RandomDatasetCSV/ascii\\", "")
    #     filename = filename.replace(".csv", "")

    #     status_dict = {
    #         "vectorizer" : "ascii",
    #         "targetCSV" : path,
    #         "SavingPath" : f"experiment/RandomDataProject/MachineLearning/RandomForest/ascii/{filename}.txt",
    #         "SavingPlotPath" : f"experiment/RandomDataProject/MachineLearning/RandomForest/ascii/{filename}.png"
    #     }
    #     RandomForestMain(status_dict)

    

    # PositionBucket[64]用 #
    # CsvPath = "CSV/RandomDatasetCSV/bucket/64/*csv"
    # csvpaths = glob.glob(CsvPath)

    # for path in csvpaths[:]:
    #     # 保存用のファイル名作成 #
    #     filename = path.replace("CSV/RandomDatasetCSV/bucket/64\\", "")
    #     filename = filename.replace(".csv", "")

    #     status_dict = {
    #         "vectorizer" : "bucket",
    #         "targetCSV" : path,
    #         "SavingPath" : f"experiment/RandomDataProject/MachineLearning/RandomForest/bucket/[64]{filename}.txt",
    #         "SavingPlotPath" : f"experiment/RandomDataProject/MachineLearning/RandomForest/bucket/[64]{filename}.png"
    #     }
    #     RandomForestMain(status_dict)
    


    # # PositionBucket[128]用 #
    # CsvPath = "CSV/RandomDatasetCSV/bucket/128/*csv"
    # csvpaths = glob.glob(CsvPath)

    # for path in csvpaths[:]:
    #     # 保存用のファイル名作成 #
    #     filename = path.replace("CSV/RandomDatasetCSV/bucket/128\\", "")
    #     filename = filename.replace(".csv", "")

    #     status_dict = {
    #         "vectorizer" : "bucket",
    #         "targetCSV" : path,
    #         "SavingPath" : f"experiment/RandomDataProject/MachineLearning/RandomForest/bucket/[128]{filename}.txt",
    #         "SavingPlotPath" : f"experiment/RandomDataProject/MachineLearning/RandomForest/bucket/[128]{filename}.png"
    #     }
    #     RandomForestMain(status_dict)
    

    # TF-IDF用 #
    # CsvPath = "CSV/RandomDatasetCSV/tfidf/*csv"
    CsvPath = "CSV/RandomDatasetCSV/tfidf/[[]3gram[]]result*"

    csvpaths = glob.glob(CsvPath)
    print(csvpaths)
    for path in csvpaths[:]:
        # 保存用のファイル名作成 #
        filename = path.replace("CSV/RandomDatasetCSV/tfidf\\", "")
        filename = filename.replace(".csv", "")

        status_dict = {
            "vectorizer" : "tfidf",
            "targetCSV" : path,
            "SavingPath" : f"experiment/RandomDataProject/MachineLearning/RandomForest/tfidf/{filename}.txt",
            "SavingPlotPath" : f"experiment/RandomDataProject/MachineLearning/RandomForest/tfidf/{filename}.png"
        }
        RandomForestMain(status_dict)