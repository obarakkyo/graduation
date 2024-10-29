"""プログラムの説明
tlshをベクトル化した値を用いて
ランダムフォレストにより分類する。
"""

"""必要なモジュールのインポート"""
import pandas as pd
import time
import matplotlib.pyplot as plt

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
    # x = df.iloc[:, :-1]
    x = df.iloc[:, 0:100] #APIだけ
    # x = df.iloc[:, 100:-1] #API以外を使う
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

     # 分類を間違えたインデックスの抽出
    misclassified_indices = y_test.index[y_test['LABEL'] != y_pred]

    # 分類を間違えたファイル名の表示
    print("Misclassified file names:")
    print(misclassified_indices)

    #重要な特徴量を可視化
    labels = x_train.columns
    importances = best_model.feature_importances_

    plt.figure(figsize = (10,6))
    plt.barh(y = range(len(importances)), width = importances)
    plt.yticks(ticks = range(len(labels)), labels = labels)
    plt.show()


if __name__ == "__main__":

    #探索対象CSVの指定
    """ASCII系"""
    # target_csv = "../CSV/dataset6CSV/ascii/2label.csv"

    """Bucket系"""
    # target_csv = "../CSV/dataset6CSV/bucket/Position_64_2label.csv"
    # target_csv = "../CSV/dataset6CSV/bucket/Position_128_2label.csv"

    "TF-IDF系"
    # target_csv = "../CSV/dataset6CSV/tfidf/max100_1gram_2label.csv"
    # target_csv = "../CSV/dataset6CSV/tfidf/max100_2gram_2label.csv"
    # target_csv = "../CSV/dataset6CSV/tfidf/max100_3gram_2label.csv"

    """Doc2Vec系"""
    target_csv = "../CSV/dataset6CSV/doc2vec/2label.csv"


    


    #グリッドリサーチによるハイパラメータの探索候補設定
    parameters = {
    'n_estimators' : [i for i in range(50, 100, 5)],
    'max_features'  : ('sqrt', 'log2', None),
    'max_depth'   : [i for i in range(20, 50, 5)],
    }

    #モデルインスタンス
    model = RandomForestClassifier(class_weight="balanced", random_state=123)

    main(target_csv, parameters, model)



    