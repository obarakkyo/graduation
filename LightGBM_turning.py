"""必要なライブラリのインポート"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import time

def main(cav_name, params):
    """データの読み込み"""
    print('csv_name = ', cav_name)
    df = pd.read_csv(csv_name, index_col=0)
    print('df.shape = ', df.shape)

    #説明変数x type(numpy.ndarray)
    x = df.iloc[:, 0:-1].values
    print('x.shape = ', x.shape)

    #目的変数y  type(numpy.ndarray)
    y = df.loc[:, ['LABEL']].values
    print('y.shape = ', y.shape)


    """訓練データと検証用データに分割"""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)



    """モデルの作成"""
    model = lgb.LGBMClassifier(
        boosting_type = 'gbdt', #決定木勾配グースティング
        objective = 'binary', #2クラス分類
        # class_weight = 'balanced',
        n_jobs = -1,
        seed = 10
    )

    #学習時のfitパラメータ設定
    fit_params = {
        'verbose' : 0, #学習中のコマンドライン出力オフ
        'early_stopping_rounds' : 10,
        'eval_metric' : 'binary_logloss',
        'eval_set' : [(x_train, y_train), (x_test, y_test)]
    }





    """グリッドサーチの開始"""
    grid = GridSearchCV(
        estimator = model,
        param_grid = params,
        cv = 5 ,
    )

    start_time = time.time()
    grid.fit(x_train, y_train, **fit_params)
    end_time = time.time() - start_time
    print(grid.best_params_)
    print(grid.best_score_)
    print('Time : ', end_time)


    #最高性能のモデルを取得し、テストデータを分類
    best_model = grid.best_estimator_ 
    y_pred = best_model.predict(x_test)
    print('bese_model\'s score = ', best_model.score(x_test, y_test))

    #混同行列を表示
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))





    


          
if __name__ == '__main__':
    """アスキーコード変換CSV"""
    # csv_name = './CSV/Binarization/grad/ascii/ascii_100report8_report.csv'
    # csv_name = './CSV/Binarization/grad/ascii/ascii_100report8_Backdoor.csv'
    # csv_name = './CSV/Binarization/grad/ascii/ascii_100report8_Infostealer.csv'
    # csv_name = './CSV/Binarization/grad/ascii/ascii_100report8_Packed.csv'
    # csv_name = './CSV/Binarization/grad/ascii/ascii_100report8_Trojan.csv'

    """TF-IDF変換CSV"""
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram1size100report.csv'
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram1size100Backddor.csv'
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram1size100Infostealer.csv'
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram1size100Packed.csv'
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram1size100Trojan.csv'

    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram2size100report.csv'
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram2size100Backddor.csv'
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram2size100Infostealer.csv'
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram2size100Packed.csv'
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram2size100Trojan.csv'

    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram3size100report.csv'
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram3size100Backddor.csv'
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram3size100Infostealer.csv'
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram3size100Packed.csv'
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram3size100Trojan.csv'

    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram4size100report.csv'
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram4size100Backddor.csv'
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram4size100Infostealer.csv'
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram4size100Packed.csv'
    # csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram4size100Trojan.csv'

    """Doc2VecのCSV"""
    # csv_name = './CSV/Binarization/grad/doc2vec/report_s100w8alpha0.1-0.001dm0seed4.csv'
    # csv_name = './CSV/Binarization/grad/doc2vec/Backdoor_s100w8alpha0.1-0.001dm0seed4.csv'
    # csv_name = './CSV/Binarization/grad/doc2vec/infostealer_s100w8alpha0.1-0.001dm0seed4.csv'
    # csv_name = './CSV/Binarization/grad/doc2vec/Packed_s100w8alpha0.1-0.001dm0seed4.csv'
    # csv_name = './CSV/Binarization/grad/doc2vec/Trojan_s100w8alpha0.1-0.001dm0seed4.csv'


    """TLSH"""
    csv_name = "./CSV/dataset1CSV/doc2vec/tlsh_csv_doc2vec_4spilit_2label.csv"


    #ハイパーパラメータの設定
    params = {
        'reg_alpha': [0.0001, 0.001, 0.01, 0.1],
        'reg_lambda': [0.001, 0.001, 0.01, 0.1,  1],
        'num_leaves': [4, 10, 20],
        'colsample_bytree': [0.1, 0.6, 1],
        # 'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        # 'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
        'min_child_samples': [2, 4, 6, 10],
    }

    main(csv_name, params)