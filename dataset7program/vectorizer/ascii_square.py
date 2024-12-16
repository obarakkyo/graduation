import pandas as pd
import numpy as np
import time

def base_ascii_vector(target_str, LogBool):
    if type(target_str) is int:
        target_str = str(target_str)
    squares = sum(ord(char) ** 2 for char in target_str)

    if LogBool:
        #対数変換して返す
        return np.log1p(squares / len(target_str))
    else: 
        return squares / len(target_str)
    


def ASCII_vectorizer(api_df, ngram, LogBool):
    result_list = []

    for i in range(api_df.shape[0]):
        vectorized_api = []
        for j in range(api_df.shape[1] - ngram + 1):
            tmp_apiname = api_df.iloc[i, j:j+ngram].values
            join_str = "".join(tmp_apiname)
            vectorized_api.append(base_ascii_vector(join_str, LogBool))
        result_list.append(vectorized_api)
    
    # データフレームに変換 #
    columns_list = [f"API{i+1}" for i in range(len(vectorized_api))]
    result_df = pd.DataFrame(result_list, index=api_df.index, columns=columns_list)
    return result_df



def ascii_square(ngram=1, LogBool=False, status_dict=None):
    print(f"[ASCII square], n_gram={ngram}, LogBool={LogBool}")

    ### データの取得 ####
    df_origin = pd.read_csv(status_dict["Origin_CSV_Path"], index_col=0)
    print(f"df_origin.shape = {df_origin.shape}")

    ### ベクトル化 ###
    api_df = df_origin.iloc[:, 0:100]
    start_time = time.time()
    vectorized_df = ASCII_vectorizer(api_df, ngram, LogBool)
    end_time = time.time()

    ### 実行時間の記録 ###
    with open(status_dict["process_time_path"], mode="w", encoding="utf-8") as f:
        print(f"Processing Time = {end_time - start_time}", file=f)
    
    ### Summaryを連結 ###
    summary_process_df = df_origin.iloc[:, 100:]
    result_df = pd.concat((vectorized_df, summary_process_df), axis=1)

        

    ### CSVとして保存 ###
    result_df.to_csv(status_dict["Save_csv_path"])







if __name__ == "__main__":
    n_gram = [1, 2, 3]
    LogBool = [False, True]

    for current_ngram in n_gram:
        for current_Bool in LogBool:
            status_dict = {
                "Origin_CSV_Path" : f"../../CSV/dataset7CSV/origin/2label.csv",
                "process_time_path" : f"../../experiment/dataset7/vectorizer_time/ascii/{current_ngram}gram_Log{current_Bool}.txt",
                "Save_csv_path" : f"../../CSV/dataset7CSV/ascii/{current_ngram}gram_Log{current_Bool}.csv"

            }
            ascii_square(current_ngram, current_Bool, status_dict)


    
    
