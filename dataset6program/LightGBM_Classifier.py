"""
LightGBM × GridSearchCV
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier
# import lightgbm as lgb
import os
import time

#混同行列、適合率、再現率、F値
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report





def main():
    """ 初期値 """
    """ ASCII """
    # vectorizer_name = "ascii"
    # n_gram = "3gram"
    # LogBoolean = True

    # レポートファイル用のパス #
    # saving_file_path = f"../experiment/dataset6/LightGBM/Log{LogBoolean}/{vectorizer_name}/{n_gram}/report.txt"
    # saving_plot_path   = f"../experiment/dataset6/LightGBM/Log{LogBoolean}/{vectorizer_name}/{n_gram}/importances.png"

    # target_csv = f"../CSV/dataset6CSV/{vectorizer_name}/{n_gram}_Log{LogBoolean}_2label.csv"

    """ Bucket """
    # bucket_len = 128
    # vectorizer_name = "bucket"
    # n_gram = "3gram"
    # LogBoolean = True

    # saving_file_path = f"../experiment/dataset6/LightGBM/Log{LogBoolean}/{vectorizer_name}/{bucket_len}/{n_gram}/report.txt"
    # saving_plot_path   = f"../experiment/dataset6/LightGBM/Log{LogBoolean}/{vectorizer_name}/{bucket_len}/{n_gram}/importances.png"

    # target_csv = f"../CSV/dataset6CSV/{vectorizer_name}/{n_gram}_PositionBucket_{bucket_len}_Log{LogBoolean}.csv"


    """Doc2Vec"""
    # saving_file_path = f"../experiment/dataset6/LightGBM/doc2vec/report.txt"
    # saving_plot_path = f"../experiment/dataset6/LightGBM/doc2vec/importances.png"
    # target_csv = "../CSV/dataset6CSV/doc2vec/2label.csv"

    """TFIDF"""
    n_gram = "3gram"

    saving_file_path = f"../experiment/dataset6/LightGBM/tfidf/{n_gram}/report.txt"
    saving_plot_path = f"../experiment/dataset6/LightGBM/tfidf/{n_gram}/importances.png"
    target_csv = f"../CSV/dataset6CSV/tfidf/max100_{n_gram}_2label.csv"



    # データの取得 #
    df = pd.read_csv(target_csv, index_col=0)
    print(f"探索対象データ = {target_csv}\n")
    print(df.head(5))

    # データの分割 #
    X = df.iloc[:, :-1]
    Y = df.loc[:, ["LABEL"]]
    print("=====X(説明変数)=====")
    print(f"X.shape = {X.shape}")
    print("=====Y(目的変数)=====")
    print(f"Y.shape = {Y.shape}")

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123, shuffle=True, stratify=Y)
    print(f"=======分割後=======")
    print(f"x_train = {x_train.shape}")
    print(f"x_test  = {x_test.shape}")
    print(f"y_train = {y_train.shape}")
    print(f"y_test  = {y_test.shape}")


    
    # モデルのインスタンス化 #
    model = LGBMClassifier(boosting_type="gbdt", class_weight="balanced")

    #探索対象パラメータの設定
    params = {
              "nestimators" : [20, 50, 100],
              "num_leaves" : [31, 50, 100],
                "learning_rate" : [0.1, 0.01],
              "max_depth" : [-1, 10, 30],
              "reg_alpha" : [0.0, 0.2],
              "reg_lambda" : [1.0, 0.8],
            }
 
    #グリッドサーチの定義 #
    gridsearch = GridSearchCV(
        estimator = model,
        param_grid = params,
        scoring = 'accuracy',
        error_score='raise',
        n_jobs=-1,
        verbose=0,
    )

    # グリッドサーチの開始 #
    start_time = time.time()
    gridsearch.fit(x_train, y_train)
    end_time = time.time()
    os.system('cls')


    with open(saving_file_path, mode="w", encoding="utf-8") as f:
        #探索時間
        total_time = end_time - start_time
        print(f"グリッドサーチ時間 = {total_time}", file=f)
        """グリッドサーチで得られた情報を取得"""
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
    plt.savefig(saving_plot_path)
    plt.show()







if __name__ == "__main__":
    main()