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







def choice_Vectorized_CSV():
    """
    学習で利用するCSVを選択する関数
    """

    """ASCII"""
    vectorizer = "ascii"
    n_gram = [1, 2, 3]
    LogBool = [True, False]
    for current_Bool in LogBool:
        for current_gram in n_gram:
            status_dict = {
                "target_csv" : f"../CSV/dataset7CSV/{vectorizer}/{current_gram}gram_Log{current_Bool}.csv",
                "saving_filepath" : f"../experiment/dataset7/LightGBM/Log{current_Bool}/{vectorizer}/{current_gram}gram/report.txt",
                "saving_plotpath" : f"../experiment/dataset7/LightGBM/Log{current_Bool}/{vectorizer}/{current_gram}gram/importtances.png",
            }
            LGBmain(status_dict)
    
    """Bucket"""
    vectorizer = "bucket"
    bucket_len = [64, 128]
    n_gram = [1, 2, 3]
    LogBool = [True, False]
    for current_len in bucket_len:
        for current_Bool in LogBool:
            for current_gram in n_gram:
                status_dict = {
                "target_csv" : f"../CSV/dataset7CSV/{vectorizer}/{current_gram}gram_Log{current_Bool}_{current_len}.csv",
                "saving_filepath" : f"../experiment/dataset7/LightGBM/Log{current_Bool}/{vectorizer}/{current_len}/{current_gram}gram/report.txt",
                "saving_plotpath" : f"../experiment/dataset7/LightGBM/Log{current_Bool}/{vectorizer}/{current_len}/{current_gram}gram/importtances.png",
                }
                LGBmain(status_dict)
    
    """Doc2Vec"""
    status_dict = {
    "target_csv" : "../CSV/dataset7CSV/doc2vec/2label.csv",
    "saving_filepath" : f"../experiment/dataset7/LightGBM/doc2vec/report.txt",
    "saving_plotpath" : f"../experiment/dataset7/LightGBM/doc2vec/importances.png",
    }
    LGBmain(status_dict)

    """tfidf"""
    vectorizer = "tfidf"
    n_gram = [1, 2, 3]
    for current_gram in n_gram:
        status_dict = {
        "target_csv" : f"../CSV/dataset7CSV/tfidf/max100_{current_gram}gram.csv",
        "saving_filepath" : f"../experiment/dataset7/LightGBM/tfidf/{current_gram}gram/report.txt",
        "saving_plotpath" : f"../experiment/dataset7/LightGBM/tfidf/{current_gram}gram/importances.png",
        }
        LGBmain(status_dict)


            


def LGBmain(status_dict):
    """
    LightGBMのメイン関数
    """
    # データの取得 #
    df = pd.read_csv(status_dict["target_csv"], index_col=0)
    print(f"探索対象データ = {status_dict['target_csv']}")

    # データの分割　#
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

    with open(status_dict["saving_filepath"], mode="w", encoding="utf-8") as f:
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
    plt.savefig(status_dict["saving_plotpath"])
    # plt.show()

if __name__ == "__main__":
    choice_Vectorized_CSV()
