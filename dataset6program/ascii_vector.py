"""
データセット4のアスキーコード変換を行うプログラム。
"""
import pandas as pd
import time

def calculate_ascii_vector(target_str: str) -> float:
    if type(target_str) is int:
        target_str = str(target_str)
    squares = sum(ord(char) ** 2 for char in target_str)
    return squares / len(target_str)

def main():
    # 対象CSVファイルをデータフレームにする#
    csv_path = "../CSV/dataset6CSV/origin/6label.csv"
    df = pd.read_csv(csv_path, index_col=0)
    print(df.head())
    print("df.shape = {}".format(df.shape))

    #ベクトル化（ASCII）
    start_time = time.time()
    vectorized_data = df.iloc[:, 0:100].applymap(calculate_ascii_vector)
    end_time = time.time()
    df.iloc[:, 0:100] = vectorized_data

    # 計測時間をファイルに保存#
    print(end_time - start_time)
    

    #CSV化
    # df.to_csv("../CSV/dataset6CSV/ascii/6label.csv")
    # print("Vectorization time is {}s".format(end_time - start_time))

if __name__ == "__main__":
    print("-----This program changes dataset6 to ascii!!-----")
    main()