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



def main(target_csv, dimention, parameters, model):
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
    )
    grid_start_time = time.time()
    gridsearch.fit(x_train, y_train)
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
    print('bese_model\'s score = ', best_model.score(x_test, y_test))

    #混同行列を表示
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))





if __name__ == '__main__':
    #初期設定
    """アスキーコード変換CSV"""
    # target_csv = './CSV/Binarization/grad/ascii/ascii_100report8_report.csv'
    # target_csv = './CSV/Binarization/grad/ascii/ascii_100report8_Backdoor.csv'
    # target_csv = './CSV/Binarization/grad/ascii/ascii_100report8_Infostealer.csv'
    # target_csv = './CSV/Binarization/grad/ascii/ascii_100report8_Packed.csv'
    # target_csv = './CSV/Binarization/grad/ascii/ascii_100report8_Trojan.csv'

    """TF-IDF変換CSV"""
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram1size100report.csv'
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram1size100Backddor.csv'
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram1size100Infostealer.csv'
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram1size100Packed.csv'
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram1size100Trojan.csv'

    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram2size100report.csv'
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram2size100Backddor.csv'
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram2size100Infostealer.csv'
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram2size100Packed.csv'
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram2size100Trojan.csv'

    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram3size100report.csv'
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram3size100Backddor.csv'
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram3size100Infostealer.csv'
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram3size100Packed.csv'
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram3size100Trojan.csv'

    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram4size100report.csv'
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram4size100Backddor.csv'
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram4size100Infostealer.csv'
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram4size100Packed.csv'
    # target_csv = './CSV/Binarization/grad/tfidf/tfidf_ngram4size100Trojan.csv'


    """Doc2VecのCSV"""
    # target_csv = './CSV/Binarization/grad/doc2vec/report_s100w8alpha0.1-0.001dm0seed4.csv'
    # target_csv = './CSV/Binarization/grad/doc2vec/Backdoor_s100w8alpha0.1-0.001dm0seed4.csv'
    # target_csv = './CSV\Binarization/grad/doc2vec/infostealer_s100w8alpha0.1-0.001dm0seed4.csv'
    # target_csv = './CSV\Binarization/grad/doc2vec/Packed_s100w8alpha0.1-0.001dm0seed4.csv'
    target_csv = './CSV\Binarization/grad/doc2vec/Trojan_s100w8alpha0.1-0.001dm0seed4.csv'
    



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

