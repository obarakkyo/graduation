"""
XGBoostのチューニングを行うファイル。
モデルにチューニングにはグリッドリサーチを採用。
"""


"""必要なライブラリのインポート"""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import time


def main(csv_path, dimention, params):
    print("対象のcsv = ", csv_path)

    """データの読み込み"""
    df = pd.read_csv(csv_path, index_col=0)
    print('df.shape = ', df.shape)
    print(df.head(1))

    #説明変数と目的変数に分割
    x = df.iloc[:, 0:dimention]
    print('x.shape = ', x.shape)
    # print(x.head(1))
    t = df.loc[:, ['LABEL']]
    print('t.shape = ', t.shape)
    # print(t.head())


    """訓練データとテストデータの分割"""
    x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, stratify=t)


    """XGBoostの読み込みとパラメータの設定"""
    model = xgb.XGBClassifier(
        booster = 'gbtree',
        objective = 'binary:logistic',
        seed = 10)

    #学習時のパラメータ指定
    fit_params = {
        'verbose' :  0,  # 学習中のコマンドライン出力
        'early_stopping_rounds': 10,  # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
        'eval_metric': 'logloss',  # early_stopping_roundsの評価指標
        'eval_set': [(x_train, y_train), (x_test, y_test)]  # early_stopping_roundsの評価指標算出用データ
        }

    gridsearch = GridSearchCV(
        estimator = model,
        param_grid = params,
        verbose = 1,
    )


    """グリッドリサーチ開始"""
    start_time = time.time()
    print('Gridresearch Start!!')
    gridsearch.fit(x_train, y_train, **fit_params)
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







if __name__ == '__main__':

    """アスキーコード変換CSV"""
    # csv_path = './CSV/Binarization/grad/ascii/ascii_100report8_report.csv'
    # csv_path = './CSV/Binarization/grad/ascii/ascii_100report8_Backdoor.csv'
    # csv_path = './CSV/Binarization/grad/ascii/ascii_100report8_Infostealer.csv'
    # csv_path = './CSV/Binarization/grad/ascii/ascii_100report8_Packed.csv'
    # csv_path = './CSV/Binarization/grad/ascii/ascii_100report8_Trojan.csv'

    """TF-IDF変換CSV"""
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram1size100report.csv'
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram1size100Backddor.csv'
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram1size100Infostealer.csv'
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram1size100Packed.csv'
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram1size100Trojan.csv'

    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram2size100report.csv'
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram2size100Backddor.csv'
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram2size100Infostealer.csv'
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram2size100Packed.csv'
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram2size100Trojan.csv'

    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram3size100report.csv'
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram3size100Backddor.csv'
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram3size100Infostealer.csv'
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram3size100Packed.csv'
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram3size100Trojan.csv'

    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram4size100report.csv'
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram4size100Backddor.csv'
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram4size100Infostealer.csv'
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram4size100Packed.csv'
    # csv_path = './CSV/Binarization/grad/tfidf/tfidf_ngram4size100Trojan.csv'


    """Doc2VecのCSV"""
    # csv_path = './CSV/Binarization/grad/doc2vec/report_s100w8alpha0.1-0.001dm0seed4.csv'
    # csv_path = './CSV/Binarization/grad/doc2vec/Backdoor_s100w8alpha0.1-0.001dm0seed4.csv'
    # csv_path = './CSV/Binarization/grad/doc2vec/infostealer_s100w8alpha0.1-0.001dm0seed4.csv'
    # csv_path = './CSV/Binarization/grad/doc2vec/Packed_s100w8alpha0.1-0.001dm0seed4.csv'
    csv_path = './CSV/Binarization/grad/doc2vec/Trojan_s100w8alpha0.1-0.001dm0seed4.csv'

    #ベクトル化の次元数
    dimention = 100

    #チューニング対象のパラメータ
    params = {
        'max_depth' : [2, 5, 10, 15],
        'n_estimators' : [25, 30],
        'reg_alpha'    : [0.0001, 0.001, 0.1, 0, 1],
        'reg_lambda'   : [0.001, 0.01, 0.1, 1],
        'learning_rate': [0.001, 0.01, 0.03, 0.1, 1.0]
    }

        

    main(csv_path, dimention, params)