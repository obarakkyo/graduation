"""
XGBoostモデルを用いて、TLSHのCSVファイルを分類する。
"""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import time

def main(csv_path: str, dimention: int, params: dict) -> None:
    print('This is main program!')

    """データの読み込み"""
    df = pd.read_csv(csv_path, index_col=0)
    print("CSV_path = ", csv_path)
    print("df.shape = ", df.shape)
    print(df.head())


    """説明変数と目的変数に分割"""
    x = df.iloc[:, 0:-1]
    t = df.loc[:, ["LABEL"]]
    print("x.shape = ", x.shape)
    print("t.shape = ", t.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, stratify=t)

    """XGBoostの読み込みとパラメータ設定"""
    model = xgb.XGBClassifier(
        booster = 'gbtree',
        objective = 'binary:logistic',
        seed = 10,
        early_stopping_rounds=10,
        eval_metric='logloss',
    )

    """学習時のパラメータ指定"""
    fit_params = {
        'verbose' :  0,                                     # 学習中のコマンドライン出力           
        'eval_set': [(x_train, y_train), (x_test, y_test)]  # early_stopping_roundsの評価指標算出用データ
    } 

    """グリッドサーチに設定"""
    gridsearch = GridSearchCV(
        estimator = model,
        param_grid = params,
        verbose = 1,
        n_jobs = -1,
        scoring='accuracy',
    )


    """グリッドリサーチ開始"""
    start_time = time.time()
    print('Gridresearch Start!!')
    gridsearch.fit(x_train, y_train.values.ravel(), **fit_params)
    end_time = time.time() - start_time
    print('グリッドサーチ時間 : ', end_time)


    """最適パラメータの出力"""
    best_params = gridsearch.best_params_
    best_score = gridsearch.best_score_
    print('Best_params = ', best_params)
    print('Best_score = ', best_score)

    """改めて最適パラメータで学習"""
    model = xgb.XGBClassifier(**gridsearch.best_params_)
    model.fit(x_train, y_train)

    #評価
    pred = model.predict(x_test)
    print(accuracy_score(y_test, pred))
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))









if __name__ == "__main__":
    """対象CSVを指定"""
    csv_path = '../CSV/anything/tlsh_csv_doc2vec_2label.csv'


    """ベクトルの次元数を指定"""
    dimention_num = 100

    """チューニング対象のパラメータ"""
    #チューニング対象のパラメータ
    params = {
        'max_depth' : [5, 10, 15, 20, 25],
        'n_estimators' : [25, 30, 50],
        'reg_alpha'    : [0.0001, 0.001, 0.1, 0, 1],
        'reg_lambda'   : [0.001, 0.01, 0.1, 1],
        'learning_rate': [0.001, 0.01, 0.03, 0.1, 1.0]
    }

    


    main(csv_path, dimention_num, params)