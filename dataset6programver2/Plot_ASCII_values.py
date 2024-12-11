"""
ASCII変換したデータセットの値をヒストグラムと
統計量を計算して出力。
"""
import pandas as pd
import matplotlib.pyplot as plt

import itertools
import statistics




def main():
    """CSVの取得"""
    # target_csv = "../CSV/dataset6CSV/ascii/1gram_LogFalse_2label.csv"
    # target_csv = "../CSV/dataset7CSV/bucket/1gram_LogFalse_64.csv"
    target_csv = "../CSV/dataset7CSV/bucket/1gram_LogFalse_128.csv"

    df = pd.read_csv(target_csv, index_col=0)
    print(f"df.shape = {df.shape}")

    """ベクトル化されたAPIデータを取得"""
    api_df = df.iloc[:, 0:100].values.tolist()
    print(type(api_df))

    """リストを1次元配列にする"""
    api_list = list(itertools.chain.from_iterable(api_df))
    print(len(api_list))

    """統計量算出"""
    print(f"max     = {max(api_list)}")
    print(f"min     = {min(api_list)}")
    print(f"mean    = {statistics.mean(api_list)}")
    print(f"median  = {statistics.median(api_list)}")


    """Plot"""
    # plt.hist(api_list)
    # plt.show()


    


if __name__ == "__main__":
    main()