"""プログラムの説明
tlshをベクトル化した値を用いて
ランダムフォレストにより分類する。
"""

"""必要なモジュールのインポート"""
import pandas as pd
import time

#ランダムフォレスト、グリッドリサーチ
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#混同行列、適合率、再現率、F値
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def main(target_csv, dimention, parameters, model) -> None:
    """データのロード"""
    df = pd.read_csv(target_csv, index_col=0)
    print('df.shape = ', df.shape)

    #説明変数
    x = df.iloc[:, 0:dimention]
    print('x.shape = ', x.shape)

    #目的変数
    t = df.loc[:, ['LABEL']]
    print('t.shape = ', t.shape)
    print() #<--改行




    """学習データの分割"""
    x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0, stratify=t)
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)



    """グリッドリサーチによる演算実行"""
    gridsearch = GridSearchCV(
        estimator = model,
        param_grid = parameters,
        scoring = 'accuracy',
        n_jobs=-1,
        verbose=0,
    )
    grid_start_time = time.time()
    gridsearch.fit(x_train, y_train.values.ravel())
    grid_end_time = time.time()
    print('GridSearch Finished!!!')
    print('Time : ',grid_end_time - grid_start_time)


    """グリッドサーチで得られた情報を取得"""
    #最適なパラメータの取得
    print('Best params : {}'.format(gridsearch.best_params_))
    print('Best Score  : {}'.format(gridsearch.best_score_))

    #最高性能のモデルを取得し、テストデータを分類
    best_model = gridsearch.best_estimator_
    y_pred = best_model.predict(x_test)
    print('best_model\'s score = ', best_model.score(x_test, y_test))

    #混同行列を表示
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    print("This program learns TLSH using random forest!!")

    # target_csv = "../CSV/anything/tlsh_csv_doc2vec_2label.csv"
    # target_csv = "../CSV/anything/tlsh_csv_doc2vec_3spilit_2label.csv"
    target_csv = "../CSV/anything/tlsh_csv_doc2vec_1spilit_2label.csv"

    #次元数の初期設定
    dimention = 100

    #グリッドリサーチによるハイパラメータの探索候補設定
    parameters = {
    'n_estimators' : [i for i in range(50, 80, 5)],
    'max_features'  : ('sqrt', 'log2', 'auto', None),
    'max_depth'   : [i for i in range(20, 50, 5)],
    }

    #モデルインスタンス
    model = RandomForestClassifier(class_weight="balanced")

    main(target_csv, dimention, parameters, model)



    