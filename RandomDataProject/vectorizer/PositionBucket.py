"""
RandomDataProject用のPositionBucketベクトル化プログラム

今のところ、N-gamの昨日は考慮していない。

ベースディレクトリ=graduation
"""

import glob #ファイルパス取得用
import time #時間計測用
import numpy as np
import pandas as pd

def base_positionbucket(target_str:str, num_bucket=64):
    changed_vector = 0           
    target_len = len(target_str) #文字列の長さ
    buket_list = [0]*num_bucket

    for i, char in enumerate(target_str):
        buket_index = (ord(char)) % num_bucket
        buket_list[buket_index] += (0.1*i+1)
    
    changed_vector = sum((i*value) for i, value in enumerate(buket_list)) / target_len
    return changed_vector



def PositionBucketMain(CSVpath:str, bucketlen) -> None:
    # CSVを取得 #
    origin_df = pd.read_csv(CSVpath, index_col=0)
    print(f"origin_df.shape = {origin_df.shape}")


    # ベクトル化 #
    start_time = time.time()
    vectorized_df = origin_df.iloc[:, 0:100].applymap(lambda x:base_positionbucket(x, bucketlen))
    end_time = time.time()

    origin_df.iloc[:, 0:100] = vectorized_df



    # 処理時間を保存 #
    file_name = CSVpath.replace("CSV/RandomDatasetCSV/origin\\", "")
    file_name = file_name.replace(".csv", "")
    with open(f"experiment/RandomDataProject/TimeLog/bucket/[{bucketlen}]{file_name}.txt", mode="w", encoding="utf-8") as timefile:
        print(f"処理時間 = {end_time - start_time}", file=timefile)
    
    # 保存 #
    origin_df.to_csv(f"CSV/RandomDatasetCSV/bucket/{bucketlen}/{file_name}.csv")


    




if __name__ == "__main__":
    target_CSV_Folder = "CSV/RandomDatasetCSV/origin/*"
    CSVpaths = glob.glob(target_CSV_Folder)
    for path in CSVpaths[:]:
        print(f"\n#################{path}のベクトル化を開始###########################")

        for bucketlen in [64, 128]:
            PositionBucketMain(CSVpath=path, bucketlen=bucketlen)