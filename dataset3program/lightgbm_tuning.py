"""
LightGBMを用いてdataset3のデータをチューニングする
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb

def create_model():
    params = {
        "boosting_type" :"gbdt", #決定木勾配ブースティング
        # "objective" : "binary",    #２クラス分類
        "objective" : "multiclass",    #多クラス分類
        "class_weight" : "balanced",
        "n_jobs" : -1,
        "random_state" : 10,
        # "early_stopping_round" : 10,
    }

    model = lgb.LGBMClassifier(**params)

    return model

def df_split_train_test(df):
    x = df.iloc[:, 0:df.shape[1]-1].values
    y = df.loc[:, ["LABEL"]].values.ravel()
    # print(x[0:1])
    # print(y[0:1])
    print('x.shape = ', x.shape)
    print('y.shape = ', y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    return x_train, x_test, y_train, y_test

def grid_search(model, cv_params, x_train, x_test, y_train, y_test):
    """グリッドサーチの実行"""

    fit_params = {
        "eval_set": [(x_test, y_test)],
        # "eval_metric": 'binary_logloss',
        "eval_metric": "multi_logloss",
    }
    
    grid = GridSearchCV(
        estimator=model,
        param_grid=cv_params,
        cv=5,
        scoring="accuracy",
    )
    
    start_time = time.time()
    # grid.fit(x_train, y_train, eval_set=[(x_test, y_test)])
    grid.fit(x_train, y_train, **fit_params)
    end_time = time.time() - start_time
    
    print(grid.best_params_)
    print(grid.best_score_)
    print('Time : ', end_time)
    
    return grid

def main():
    print("このプログラムはLightGBMを用いて、dataset3を分類します。\n")

    """初期設定変数"""
    # target_csv = "../CSV/dataset3CSV/ascii/ascii_2label.csv"
    target_csv = "../CSV/dataset3CSV/doc2vec/6label.csv"

    cv_params = {
        'reg_alpha': [0.0001, 0.001, 0.01, 0.1],
        'reg_lambda': [0.001, 0.01, 0.1, 1],
        'num_leaves': [4, 10, 20],
        'colsample_bytree': [0.1, 0.6, 1],
        'min_child_samples': [2, 4, 6, 10],
    }



    """データの読み込み"""
    df = pd.read_csv(target_csv, index_col=0)
    print("df.shape = {}".format(df.shape))

    """訓練データとテストデータに分割"""
    x_train, x_test, y_train, y_test = df_split_train_test(df)

    """モデルの構築"""
    model = create_model()

    """グリッドサーチ開始"""
    grid = grid_search(model, cv_params, x_train, x_test, y_train, y_test)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)
    
    print('best_model\'s score = ', best_model.score(x_test, y_test))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    

    
    

if __name__ == "__main__":
    main()