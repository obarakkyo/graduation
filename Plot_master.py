"""あるモデルにおけるパラメータの変化における正解率の変化をPlotして
   パラメータチューニング対象の範囲を可視化してみる"""

"""必要なライブラリのインポート"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import validation_curve, train_test_split
import matplotlib.pyplot as plt



#訓練データと検証用データを作成して返す
#引数 ①対象csvの相対パス ②特徴量の数 default=100
def create_train_test(csv_name, dimention):
    """対象のデータを読み込んで表示"""
    df = pd.read_csv(csv_name, index_col=0)
    print('df.shape = ', df.shape)

    #訓練データと検証用データに分割
    x = df.iloc[:, 0:dimention]
    t = df.loc[:, ['LABEL']]
    print('訓練データのshape   = ', x.shape)
    print('テストデータのshape = ', t.shape)
    print()
    x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0, stratify=t)
    print('x_train.shape = ', x_train.shape)
    print('x_test.shape  = ', x_test.shape)
    print('y_train.shape = ', y_train.shape)
    print('y_test.shape  = ', y_test.shape)
    print()

    print(x_train.head())

    return x_train, x_test, y_train, y_test




def main(cv_params, params_scales, model, x_train, y_train, fit_params):
    """validation_curveの実行"""

    scoring = 'neg_log_loss'
    for i, (k, v) in enumerate(cv_params.items()):
        train_scores, valid_scores = validation_curve(
            estimator = model,
            X = x_train,
            y = y_train,
            param_name = k,
            param_range = v,
            fit_params=fit_params,
            cv = 5,
            n_jobs = -1,
        )

        print(train_scores)

        #学習データに対するスコアの平均+-標準偏差を算出
        train_mean   = np.mean(train_scores, axis=1)
        train_std    = np.std(train_scores, axis=1)
        train_center = train_mean
        train_high   = train_mean + train_std
        train_low    = train_mean - train_std

        #テストデータに対するスコアの平均+-標準偏差を算出
        vaild_mean   = np.mean(valid_scores, axis = 1)
        valid_std    = np.std(valid_scores, axis=1)
        valid_center = vaild_mean
        valid_high   = valid_center + valid_std
        valid_low    = valid_center - valid_std

        #train_scoresをプロット
        plt.plot(v, train_center, color='blue', marker='o', markersize=5,label='training score')
        plt.fill_between(v, train_high, train_low, alpha=0.15, color='blue')

        #valid_scoresをプロット
        plt.plot(v, valid_center, color='green', linestyle='--', marker='o', markersize=5, label='validation scores')
        plt.fill_between(v, valid_high, valid_low, alpha=0.15, color='green', )
        
        #スケールをparams_scalesに合わせて変更
        plt.xscale(params_scales[k])

        #軸ラベルおよび凡例の指定
        plt.xlabel(k)
        plt.ylabel(scoring)
        plt.legend(loc='lower right')

        #グラフを描画
        plt.show()










if __name__ == "__main__":

    #対象とするcsvの相対pathを格納
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
    csv_name = './CSV/Binarization/grad/doc2vec/report_s100w8alpha0.1-0.001dm0seed4.csv'
    # csv_name = './CSV/Binarization/grad/doc2vec/Backdoor_s100w8alpha0.1-0.001dm0seed4.csv'
    # csv_name = './CSV/Binarization/grad/doc2vec/infostealer_s100w8alpha0.1-0.001dm0seed4.csv'
    # csv_name = './CSV/Binarization/grad/doc2vec/Packed_s100w8alpha0.1-0.001dm0seed4.csv'
    # csv_name = './CSV/Binarization/grad/doc2vec/Trojan_s100w8alpha0.1-0.001dm0seed4.csv'

    #csvファイルの特徴量の数を指定
    dimention = 100

    #訓練データと検証用データを作成 　※自作関数create_train_testを使用
    x_train, x_test, y_train, y_test = create_train_test(csv_name, dimention)

    """---------------ランダムフォレストの時--------------------"""
    # model = RandomForestClassifier(verbose=0, class_weight='balanced', max_features='sqrt')

    # #ハイパーパラメータと探索範囲のリストを辞書する
    # cv_params = {
    #     'n_estimators' : [i for i in range(10, 200, 2)],
    #     'max_depth'    : [i for i in range(5, 100, 2)],
    #     'min_samples_split' : [i for i in range(1, 15)],
    # }

    # #ハイパーパラメータのスケールを示す辞書
    # params_scales = {
    #     'n_estimators' : 'linear',
    #     'max_depth'    : 'linear',
    #     'min_samples_split' : 'linear',
    # }

    # main(cv_params, params_scales, model, x_train, y_train)
    """-------------------------------------------------------"""






    """--------------------XGBoostの時-------------------------"""
    # model = xgb.XGBClassifier(
    #     booster = 'gbtree',
    #     objective = 'binary:logistic',
    #     seed = 10,
    # )

    # fit_params = {
    #     'verbose' : 0, #学習中のコマンドライン出力  
    #     'early_stopping_rounds' : 10, #学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
    #     'eval_metric' : 'logloss', 
    #     'eval_set'    : [(x_train, y_train), (x_test, y_test)]   #評価指標
    # }

    # cv_params = {
    #     'max_depth'    : [i for i in range(2, 100, 2)],
    #     'n_estimators' : [i for i in range(10, 200, 10)],
    #     'reg_alpha'    : [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
    #     'reg_lambda'    : [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
    #     'learning_rate': [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
    #  }

    # params_scales = {
    #     'max_depth' : 'linear',
    #     'n_estimators' : 'linear',
    #     'reg_alpha'    : 'log',
    #     'reg_lambda'   : 'log',
    #     'learning_rate': 'log',
    # }
    # main(cv_params, params_scales, model, x_train, y_train, fit_params)
    """------------------------------------------------------------"""





    """-------------------------LightGBMの時---------------------------"""
    model = lgb.LGBMClassifier(
        boosting_type = 'gbdt', #決定木勾配ブースティング
        objective = 'binary', #2クラス分類
        class_weight = "balanced",
        seed = 10,
    )

    model.set_params(num_leaves=2, min_child_samples=5)

   #学習時のfitパラメータ設定
    fit_params = {
        'verbose' : 0, #学習中のコマンドライン出力オフ
        'early_stopping_rounds' : 10,
        'eval_metric' : 'binary_logloss',
        'eval_set' : [(x_train, y_train), (x_test, y_test)]
    }

    #ハイパーパラメータの設定
    cv_params = {
        'reg_alpha': [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
        'reg_lambda': [0, 0.0001, 0.001, 0.01, 0.1, 0.3, 1],
        'num_leaves': [2, 4, 8, 16, 32, 64, 96],
        'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
        'min_child_samples': [0, 2, 5, 10, 20, 30, 40, 50],
    }

    params_scales = {
        'reg_alpha'  : 'log',
        'reg_lambda' : 'log',
        'num_leaves' : 'linear',
        'colsample_bytree' : 'linear',
        'subsample' : 'linear',
        'subsample_freq' : 'linear',
        'min_child_samples' : 'linear',
    }  
    main(cv_params, params_scales, model, x_train, y_train, fit_params)
    """------------------------------------------------------------------"""


        



