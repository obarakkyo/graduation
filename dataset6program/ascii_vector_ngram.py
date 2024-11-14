"""
ASCIIベクトル化にN-gram機能を追加したプログラム。

"""
import pandas as pd
import time
from tqdm import tqdm
import numpy as np

def create_column_name(columns_len, n_gram):
    columns_list = []
    for i in range(columns_len):
        columns_str = f"API{i+1}"
        for j in range(n_gram-1):
            columns_str += f"_{i+j+2}"
        columns_list.append(columns_str)
    return columns_list





def calculate_ascii_vector(target_str: str, LogTrans=False) -> float:
    if type(target_str) is int:
        target_str = str(target_str)
    squares = sum(ord(char) ** 2 for char in target_str)

    if LogTrans:
        #対数変換して返す
        return np.log1p(squares / len(target_str))
    else: 
        return squares / len(target_str)
        



def ascii_ngram_vectorizer(target_df, n_gram, LogTrans=False):
    result_list = []

    for i in tqdm(range(target_df.shape[0])):
        vectorized_api100 = []
        for j in range(target_df.shape[1] - n_gram + 1):
            get_apiname = target_df.iloc[i, j:j+n_gram].values
            join_str = "".join(get_apiname)
            vectorized_api100.append(calculate_ascii_vector(join_str, LogTrans))
        result_list.append(vectorized_api100)
    
    print(len(result_list), len(result_list[0]))

    #データフレームに変換
    columns_name_list = create_column_name(len(result_list[0]),n_gram=n_gram)
    return_df = pd.DataFrame(result_list, index=target_df.index, columns=columns_name_list)
    return return_df




def main():
    # 設定変数 #
    n_gram = 3
    LogTrans = True

    # 対象CSVファイルをデータフレームにする#
    csv_path = "../CSV/dataset6CSV/origin/2label.csv"
    df = pd.read_csv(csv_path, index_col=0)
    print(df.head())
    print("df.shape = {}".format(df.shape))


    # ベクトル化 #
    start_time = time.time()
    vectorized_df = ascii_ngram_vectorizer(df.iloc[:, 0:100], n_gram=n_gram, LogTrans=LogTrans)
    end_time   = time.time()

    # ファイルに計測時間を書き込む #
    saving_file_path = f"../experiment/vectorizer_time/ascii/Log{LogTrans}/{n_gram}gram/time.txt"
    with open(saving_file_path, mode="w", encoding="utf-8") as f:
        print(f"ベクトル化時間 = {end_time - start_time}", file=f)


     ### 後ろのSummary情報を抜き出して連結 ###
    summary_process_df = df.iloc[:, 100:]
    save_df = pd.concat((vectorized_df, summary_process_df), axis=1)
    
    ### 保存 ###
    # save_df.to_csv(f"../CSV/dataset6CSV/ascii/{n_gram}gram_Log{LogTrans}2label.csv")

if __name__ == "__main__":
    main()