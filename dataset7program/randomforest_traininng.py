"""
データセット7を対象にした，ランダムフォレストによる分類．
"""

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




def RMmain(target_csv, saving_filepath, saving_plotpath):
    print(f"target_csv = {target_csv}")



    """データのロード"""
    df = pd.read_csv(target_csv, index_col=0)
    print('df.shape = ', df.shape)

    #説明変数
    x = df.iloc[:, :-1]
    # x = df.iloc[:, 0:100] #APIだけ
    # x = df.iloc[:, 0:99] #APIだけ【2gram】
    # x = df.iloc[:, 0:98] #APIだけ【3gram】
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


    """グリッドリサーチによるハイパラメータの探索候補設定"""
    parameters = {
        'n_estimators' : [i for i in range(50, 100, 5)],
        'max_features'  : ('sqrt', 'log2', None),
        'max_depth'   : [i for i in range(20, 50, 5)],
    }

    #モデルインスタンス
    model = RandomForestClassifier(class_weight="balanced", random_state=123)



    """グリッドリサーチによる演算実行"""
    gridsearch = GridSearchCV(
        estimator = model,
        param_grid = parameters,
        scoring = 'accuracy',
        cv=5,
        n_jobs=-1,
        verbose=0,
    )

    grid_start_time = time.time()
    gridsearch.fit(x_train, y_train.values.ravel())
    grid_end_time = time.time()

    with open(saving_filepath, mode="w", encoding="utf-8") as f:
        print('GridSearch Finished!!!', file=f)
        print('Time : ',grid_end_time - grid_start_time, file=f)


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
    plt.savefig(saving_plotpath)
    # plt.show()


def choice_Vectorized_CSV():
    saving_filepath = ""

    """ASCII"""
    # vectorizer = "ascii"
    # n_gram = [1, 2, 3]
    # LogBool = [True, False]
    # for current_gram in n_gram:
    #     for current_Bool in LogBool:
    #         target_csv = f"../CSV/dataset7CSV/{vectorizer}/{current_gram}gram_Log{current_Bool}.csv"
    #         saving_filepath = f"../experiment/dataset7/RandomForest/Log{current_Bool}/{vectorizer}/{current_gram}gram/report.txt"
    #         saving_plotpath = f"../experiment/dataset7/RandomForest/Log{current_Bool}/{vectorizer}/{current_gram}gram/importtances.png"
    #         RMmain(target_csv=target_csv, saving_filepath=saving_filepath, saving_plotpath=saving_plotpath)


    """Bucket"""
    # vectorizer="bucket"
    # n_gram = [1, 2, 3]
    # LogBool = [False, True]
    # bucket_len = [64, 128]
    # for current_ngram in n_gram:
    #     for current_Bool in LogBool:
    #         for current_len in bucket_len:
    #             target_csv = f"../CSV/dataset7CSV/{vectorizer}/{current_ngram}gram_Log{current_Bool}_{current_len}.csv"
    #             saving_filepath = f"../experiment/dataset7/RandomForest/Log{current_Bool}/{vectorizer}/{current_len}/{current_ngram}gram/report.txt"
    #             saving_plotpath = f"../experiment/dataset7/RandomForest/Log{current_Bool}/{vectorizer}/{current_len}/{current_ngram}gram/importances.png"
    #             RMmain(target_csv=target_csv, saving_filepath=saving_filepath, saving_plotpath=saving_plotpath)
    

    """Tf-idf"""
    vectorizer = "tfidf"
    n_gram = [1 ,2, 3]
    for current_ngram in n_gram:
        target_csv = f"../CSV/dataset7CSV/tfidf/max100_{current_ngram}gram.csv"
        saving_filepath = f"../experiment/dataset7/RandomForest/tfidf/{current_ngram}gram/report.txt"
        saving_plotpath = f"../experiment/dataset7/RandomForest/tfidf/{current_ngram}gram/importances.png"
        RMmain(target_csv=target_csv, saving_filepath=saving_filepath, saving_plotpath=saving_plotpath)

    """Doc2Vec"""
    # target_csv = "../CSV/dataset7CSV/doc2vec/2label.csv"
    # saving_filepath = "../experiment/dataset7/RandomForest/doc2vec/report.txt"
    # saving_plotpath = "../experiment/dataset7/RandomForest/doc2vec/importances.png"
    # RMmain(target_csv=target_csv, saving_filepath=saving_filepath, saving_plotpath=saving_plotpath)





if __name__ == "__main__":
    choice_Vectorized_CSV()