"""
PositionBucket
"""
import pandas as pd
import numpy as np
import time

def base_bucket(target_str, scale_num=1, bucket_len=128, LogBool=False):
    changed_vector = 0
    bucket_index = 0
    target_len = len(target_str)
    bucket_list = [0]*bucket_len

    for i, char in enumerate(target_str):
        bucket_index = (ord(char)) % bucket_len
        bucket_list[bucket_index] += (0.1*i) + 1
    
    changed_vector = sum((scale_num*i*value) for i , value in enumerate(bucket_list)) / target_len

    if LogBool:
        return np.log10(changed_vector)
    else:
        return changed_vector


def Bucket_vectorizer(api_df, n_gram=1, LogBool=False, Bucket_len=128):
    result_list = []
    for i in range(api_df.shape[0]):
        vectorized_api = []
        for j in range(api_df.shape[1]-n_gram + 1):
            connect_str = api_df.iloc[i, j:j+n_gram].values
            join_str = "".join(connect_str)
            vectorized_api.append(base_bucket(join_str, scale_num=1, bucket_len=Bucket_len, LogBool=LogBool))
        result_list.append(vectorized_api)
    columns_list = [f"API{i+1}" for i in range(len(vectorized_api))]
    result_df = pd.DataFrame(result_list, index=api_df.index, columns=columns_list)
    return result_df




def position_bucekt(n_gram, LogBool, Bucket_len, status_dict):
    print(f"[PositionBucket], n_gram={n_gram}, LogBool={LogBool}, Buclet_len={Bucket_len}")

    ### データの取得 ###
    df_origin = pd.read_csv("../../CSV/dataset7CSV/origin/2label.csv", index_col=0)
    # print(f"・shape = {df_origin.shape}\n")

    ### ベクトル化 ###
    api_df = df_origin.iloc[:, 0:100]
    start_time = time.time()
    vectorized_df = Bucket_vectorizer(api_df, n_gram, LogBool, Bucket_len)
    end_time   = time.time()
    print(f"実行時間 = {end_time - start_time}s")

    ### 実行時間を記録 ###
    with open(status_dict["process_time_path"], mode="w", encoding="utf-8") as f:
        print(f"Processing Time = {end_time - start_time}", file=f)

    ### Summaryを連結 ###
    summary_df = df_origin.iloc[:, 100:]
    result_df = pd.concat((vectorized_df, summary_df), axis=1)

    ### CSVとして保存 ###
    result_df.to_csv(status_dict["Save_csv_path"])





if __name__ == "__main__":
    ### 初期設定 ###
    n_gram = [1, 2, 3]
    LogBool = [True, False]
    Bucket_len = [64, 128]
    status_dict = {
        "process_time_path" : f"../../experiment/dataset7/vectorizer_time/bucket/{n_gram}gram_Log{LogBool}_{Bucket_len}.txt",
        "Save_csv_path" : f"../../CSV/dataset7CSV/bucket/{n_gram}gram_Log{LogBool}_{Bucket_len}.csv"
    }
    
    # position_bucekt(n_gram, LogBool, Bucket_len, status_dict)


    for current_ngram in n_gram:
        for curret_bool in LogBool:
            for current_len in Bucket_len:
                status_dict = {
                    "process_time_path" : f"../../experiment/dataset7/vectorizer_time/bucket/{current_ngram}gram_Log{curret_bool}_{current_len}.txt",
                    "Save_csv_path" : f"../../CSV/dataset7CSV/bucket/{current_ngram}gram_Log{curret_bool}_{current_len}.csv"
                }
                position_bucekt(current_ngram, curret_bool, current_len, status_dict)

