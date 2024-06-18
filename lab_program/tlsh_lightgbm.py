"""
TLSHのCSVを読み込んで、LightGBMを用いた分類を行う。
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import time

"""データを読み込み、分割する関数"""
def download_and_split_dataset(target_csv: str) -> None:
    #データの読み込み
    csv_data = pd.read_csv(target_csv, index_col=0)
    print(csv_data.head())
    print(type(csv_data), csv_data.shape)

    #説明変数と目的変数に分割
    x = csv_data.iloc[:, 0:csv_data.shape[1]-1].values
    y = csv_data.loc[:, ["LABEL"]].values.ravel()
    print(x[0:1])
    print(y[0:1])
    print('x.shape = ', x.shape)
    print('y.shape = ', y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    return x_train, x_test, y_train, y_test

def create_model_func():
    model = lgb.LGBMClassifier(
        boosting_type = 'gbdt', #決定木勾配ブースティング
        objective = 'binary',    #２クラス分類
        class_weight= 'balanced',
        n_jobs= -1,
        seed = 10,
        early_stopping_rounds = 10,
        # eval_metric = 'bibary_logloss',
    )
    return model


def grid_search(model, params, x_train, x_test, y_train, y_test):
    """グリッドサーチの実行"""
    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=5,
        scoring="accuracy",
    )
    
    start_time = time.time()
    # grid.fit(x_train, y_train, eval_set=[(x_test, y_test)])
    grid.fit(x_train, y_train)
    end_time = time.time() - start_time
    
    print(grid.best_params_)
    print(grid.best_score_)
    print('Time : ', end_time)
    
    return grid


if __name__ == "__main__":
    # target_csv = "../CSV/anything/tlsh_csv_doc2vec_2label.csv"
    target_csv = "../CSV/dataset1CSV/doc2vec/tlsh_csv_doc2vec_4spilit_2label.csv"

    x_train, x_test, y_train, y_test = download_and_split_dataset(target_csv)


    #モデルの作成
    model = create_model_func()

    #ハイパーパラメータの設定
    params = {
        'reg_alpha': [0.0001, 0.001, 0.01, 0.1],
        'reg_lambda': [0.001, 0.01, 0.1, 1],
        'num_leaves': [4, 10, 20],
        'colsample_bytree': [0.1, 0.6, 1],
        'min_child_samples': [2, 4, 6, 10],
    }

    grid = grid_search(model, params, x_train, x_test, y_train, y_test)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)
    
    print('best_model\'s score = ', best_model.score(x_test, y_test))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))




