import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import os
import time

#混同行列、適合率、再現率、F値
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report



def SVMmain(status_dict):
    """
    SVMのメイン関数。
    """
    
    # データセットのロード #
    df = pd.read_csv(status_dict["target_csv"], index_col=0)
    print(f"探索対象のデータ = {status_dict['target_csv']}")

    # データの分割 #
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    print(f"X.shape = {X.shape}")
    print(f"Y.shape = {Y.shape}")
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123, stratify=Y)
    print(f"=======分割後=======")
    print(f"x_train = {x_train.shape}")
    print(f"x_test  = {x_test.shape}")
    print(f"y_train = {y_train.shape}")
    print(f"y_test  = {y_test.shape}")

    # SVMのモデルの定義 #
    model = SVC(random_state=123)

    # ハイパーパラメータの候補 #
    param_grid = {
        'C' : [0.1, 1],
        'kernel' : ['linear'],
        'gamma' : ['scale'],
    }

    # グリッドサーチの定義 #
    grid_search = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        scoring = "accuracy",
        cv = 5,
        n_jobs = -1,
        verbose = 1,
    )

    # グリッドサーチの開始 #
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    end_time = time.time()
    print(f"グリッドサーチ時間 = {end_time - start_time}")

    # 最適なモデルでテストセットを予測 #
    y_pred = grid_search.best_estimator_.predict(x_test)

    # 正解率を計算 #
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Best Parameters : {grid_search.best_params_}")
    print(f"Accuracy : {accuracy}")

    #混同行列を表示
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))










def choice_Vectorized_CSV():
    """
    学習で利用するCSVを選択する関数
    """

    """ASCII"""
    vectorizer = "ascii"
    n_gram = [1]
    LogBool = [False]
    for current_Bool in LogBool:
        for current_gram in n_gram:
            status_dict = {
                "target_csv" : f"../CSV/dataset7CSV/{vectorizer}/{current_gram}gram_Log{current_Bool}.csv",
                "saving_filepath" : f"../experiment/dataset7/SVM/Log{current_Bool}/{vectorizer}/{current_gram}gram/report.txt",
                "saving_plotpath" : f"../experiment/dataset7/SVM/Log{current_Bool}/{vectorizer}/{current_gram}gram/importtances.png",
            }
            SVMmain(status_dict)








if __name__ == "__main__":
    choice_Vectorized_CSV()