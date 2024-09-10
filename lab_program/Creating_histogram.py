"""
ベクトル化したデータセットの数値が
どのように分布しているのかを表示してみる。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def main():
    print("##########START##########")

    ### データセットの取り込み ###
    csv_path = "../CSV/dataset4CSV/ascii/2labelWithoutSummary.csv"   #ASCII
    # csv_path = "../CSV/dataset4CSV/doc2vec/2labelWithoutSummary.csv" #Doc2Vec
    # csv_path = "../CSV\dataset4CSV/buket/Simplevector_2label_WithoutSummary.csv"
    # csv_path = "../CSV\dataset4CSV/buket/NexttoVector_2label_WithoutSummary.csv"
    # csv_path = "../CSV/dataset4CSV/buket/StrpositionVector_2label_WithoutSummary.csv"
    # csv_path = "../CSV/dataset4CSV/buket/Scale_StrpositionVector_2label_WithoutSummary.csv"


    df = pd.read_csv(csv_path, index_col=0)
    print("df.shape = {}".format(df.shape))

    ###　数値データだけ取り出す ###
    np_datas = df.iloc[:, 0:-1].to_numpy()
    print("np.datas.shape = {}".format(np_datas.shape))

    ### ヒストグラム用にリシェイプ　###
    x = np_datas.shape[0]
    y = np_datas.shape[1]
    reshape_datas = np_datas.flatten()
    print("reshape_datas.shape = {}".format(reshape_datas.shape))

    ### ヒストグラムとして表示　###
    fig, ax = plt.subplots()
    ax.hist(reshape_datas, bins=80)

    # plt.savefig("../pictures/dataset4_scale_strposition_buket.png")
    plt.show()

    ### 統計量を算出 ###
    data_max = np.max(reshape_datas)
    data_min = np.min(reshape_datas)
    data_mean = np.mean(reshape_datas)
    data_median = np.median(reshape_datas)
    print("最大値 = {}".format(data_max))
    print("最小値 = {}".format(data_min))
    print("平均値 = {}".format(data_mean))
    print("中央値 = {}".format(data_median))




    print("###########END###########")

if __name__ == "__main__":
    main()