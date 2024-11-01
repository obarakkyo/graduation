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

def main(target_csv, parameters, model) -> None:
    """データのロード"""
    df = pd.read_csv(target_csv, index_col=0)
    print('df.shape = ', df.shape)

    #説明変数
    x = df.iloc[:, 0:-1]
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

    """dataset1"""
    # target_csv = "../CSV/dataset1CSV/ascii/tlsh_ascii_4split_2label.csv"
    # target_csv = "../CSV/dataset1CSV/ascii/tlsh_ascii_3split_2label.csv"
    # target_csv = "../CSV/dataset1CSV/ascii/tlsh_ascii_2split_2label.csv"
    # target_csv = "../CSV/dataset1CSV/ascii/tlsh_ascii_1split_2label.csv"

    # target_csv = "../CSV/dataset1CSV/ascii/tlsh_AsciiSigmoid_4split_2label.csv"
    # target_csv = "../CSV/dataset1CSV/doc2vec/tlsh_csv_doc2vec_4spilit_18dimention_2label.csv"
    # target_csv = "../CSV/dataset1CSV/doc2vec/tlsh_csv_doc2vec_4spilit_4dimention_2label.csv"

    """dataset2"""
    # target_csv = "../CSV/dataset2CSV/ascii/tlsh_ascii_1split_2label.csv"
    # target_csv = "../CSV/dataset2CSV/ascii/tlsh_ascii_1split_6label.csv"
    # target_csv = "../CSV/dataset2CSV/ascii/tlsh_ascii_2split_2label.csv"
    # target_csv = "../CSV/dataset2CSV/ascii/tlsh_ascii_2split_6label.csv"
    # target_csv = "../CSV/dataset2CSV/ascii/tlsh_ascii_3split_2label.csv"
    # target_csv = "../CSV/dataset2CSV/ascii/tlsh_ascii_3split_6label.csv"
    # target_csv = "../CSV/dataset2CSV/ascii/tlsh_ascii_4split_2label.csv"
    # target_csv = "../CSV/dataset2CSV/ascii/tlsh_ascii_4split_6label.csv"

    # target_csv = "../CSV/dataset2CSV/doc2vec/tlsh_doc2vec_1split_72dim_2label.csv"
    # target_csv = "../CSV/dataset2CSV/doc2vec/tlsh_doc2vec_2split_36dim_2label.csv"
    # target_csv = "../CSV/dataset2CSV/doc2vec/tlsh_doc2vec_3split_24dim_2label.csv"
    # target_csv = "../CSV/dataset2CSV/doc2vec/tlsh_doc2vec_4split_18dim_2label.csv"

    # target_csv = "../CSV/dataset2CSV/doc2vec/tlsh_doc2vec_1split_72dim_6label.csv"
    # target_csv = "../CSV/dataset2CSV/doc2vec/tlsh_doc2vec_2split_36dim_6label.csv"
    # target_csv = "../CSV/dataset2CSV/doc2vec/tlsh_doc2vec_3split_24dim_6label.csv"
    target_csv = "../CSV/dataset2CSV/doc2vec/tlsh_doc2vec_4split_18dim_6label.csv"
    

    



    #グリッドリサーチによるハイパラメータの探索候補設定
    parameters = {
    'n_estimators' : [i for i in range(50, 100, 5)],
    'max_features'  : ('sqrt', 'log2', None),
    'max_depth'   : [i for i in range(20, 50, 5)],
    }

    #モデルインスタンス
    model = RandomForestClassifier(class_weight="balanced", random_state=123)

    main(target_csv, parameters, model)



    