"""
RandomDataProject用のASCIIベクトル化プログラム

今のところ、N-gamの昨日は考慮していない。

ベースディレクトリ=graduation
"""

import glob
import time
import pandas as pd
import numpy as np


def base_ascii_vector(target_str: str) -> float:
    """
    ASCIIのベース関数
    """
    if type(target_str) is int:
        target_str = str(target_str)
    squares = sum(ord(char) ** 2 for char in target_str)
    return squares / len(target_str)


def AsciiMain(CSVpath):
    """
    メイン関数
    """
    # CSVを取得 #
    origin_df = pd.read_csv(CSVpath, index_col=0)
    print(f"origin_df.shape = {origin_df.shape}")

    # ベクトル化 #
    start_time = time.time()
    vectorized_df = origin_df.iloc[:, 0:100].applymap(base_ascii_vector)
    end_time = time.time()

    origin_df.iloc[:, 0:100] = vectorized_df

    # 処理時間を保存 #
    file_name = CSVpath.replace("CSV/RandomDatasetCSV/origin\\", "")
    file_name = file_name.replace(".csv", "")
    with open(f"experiment/RandomDataProject/TimeLog/ascii/{file_name}.txt", mode="w", encoding="utf-8") as timefile:
        print(f"処理時間 = {end_time - start_time}", file=timefile)
    
    # 保存 #
    origin_df.to_csv(f"CSV/RandomDatasetCSV/ascii/{file_name}.csv")


if __name__ == "__main__":
    target_CSV_Folder = "CSV/RandomDatasetCSV/origin/*"
    CSVpaths = glob.glob(target_CSV_Folder)
    for path in CSVpaths[:]:
        print(f"\n#################{path}のベクトル化を開始###########################")
        AsciiMain(path)