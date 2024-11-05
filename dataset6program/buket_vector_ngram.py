"""
Bucket配列にNgramを導入したベクトル化手法
"""


import pandas as pd
import numpy as np
import time
from tqdm import tqdm


def create_column_name(columns_len, n_gram):
    columns_list = []
    for i in range(columns_len):
        columns_str = f"API{i+1}"
        for j in range(n_gram-1):
            columns_str += f"_{i+j+2}"
        columns_list.append(columns_str)
    return columns_list



# 文字列のi番目も考慮したものを加算 #
def position_buket(target_str:str, scale_num=0.01, num_buket=64):
    changed_vector = 0           
    target_len = len(target_str) #文字列の長さ
    buket_list = [0]*num_buket

    for i, char in enumerate(target_str):
        buket_index = (ord(char)) % num_buket
        buket_list[buket_index] += (0.1*i+1)
    
    changed_vector = sum((scale_num*i*value) for i, value in enumerate(buket_list)) / target_len
    return changed_vector


def ngram_position_bucket(target_df, n_gram=2, num_bucket=128):
    result_list = []
    for i in tqdm(range(target_df.shape[0])):
        vectorized_api100  = []
        for j in range(target_df.shape[1]-n_gram + 1):
            connect_str = target_df.iloc[i, j:j+n_gram].values
            join_str = "".join(connect_str)
            vectorized_api100.append(position_buket(join_str, scale_num=0.01, num_buket=num_bucket))
        result_list.append(vectorized_api100)
    columns_list = create_column_name(len(vectorized_api100), n_gram)
    return_df = pd.DataFrame(result_list, index=target_df.index, columns=columns_list)
    return return_df
            

def main():
    ### 設定変数 ###
    n_gram = 3
    num_bucket = 128

    ### 対象CSVの指定 ###
    target_csv_path = "../CSV/dataset6CSV/origin/2label.csv"
    df = pd.read_csv(target_csv_path, index_col=0)
    print(f"CSV_path = {target_csv_path}\ndf.shape = {df.shape}")

    ### ベクトル化 ###
    api_df = df.iloc[:, 0:100]
    start_time = time.time()
    vectorized_df = ngram_position_bucket(target_df=api_df, n_gram=n_gram, num_bucket=num_bucket)
    end_time = time.time()

    ### 後ろのSummary情報を抜き出して連結 ###
    summary_process_df = df.iloc[:, 100:]
    save_df = pd.concat((vectorized_df, summary_process_df), axis=1)
    
    ### 保存 ###
    save_df.to_csv(f"../CSV/dataset6CSV/bucket/{n_gram}gram_PositionBucket_{num_bucket}.csv")

if __name__ == "__main__":
    main()