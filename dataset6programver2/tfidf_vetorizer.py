"""
データセット6を対象に，すべてのAPIを考慮したベクトル化を行う．

１．データセットの取得
　データセットのパスを指定し，全ファイルの相対パスを取得する．
"""
import json
import time
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

summary_key_lists = ["command_line", 
                    "connects_host",
                    "connects_ip",
                    "directory_created",
                    "directory_enumerated",
                    "directory_removed",
                    "dll_loaded",
                    "downloads_file",
                    "fetches_url",
                    "file_copied",
                    "file_created",
                    "file_deleted",
                    "file_exists",
                    "file_failed",
                    "file_moved",
                    "file_opened",
                    "file_read",
                    "file_recreated",
                    "file_written",
                    "guid",
                    "mutex",
                    "regkey_deleted",
                    "regkey_opened",
                    "regkey_read",
                    "regkey_written",
                    "resolves_host",
                    "tls_master",
                    "wmi_query"]



# データセットからAPI、Index情報を抜き出す。#
def get_API_Index_from_file(_file_paths:list) -> list:
    api_list = []
    index_list = []
    all_summary_list = []

    for file_path in _file_paths[:]:
        tmp_string = ""
        tmp_summary=[]
        index_list.append(file_path.replace("../custom_datasets/dataset_6\\", ""))
        with open(file_path, mode="r") as f:
            f_json = json.load(f)
            for _, value in enumerate(f_json["all_api"]):
                tmp_string += str(value) + " "
            api_list.append(tmp_string)

            tmp_summary = get_summaryinfo(f_json)
            all_summary_list.append(tmp_summary)
    return api_list, index_list, all_summary_list


# APIリストでTF-IDFを学習させる #
def train_tfidf(all_api:list, n_gram=1, max_features=100, index_list = []):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(n_gram, n_gram))
    vectorizer.fit(all_api)

    tmp_np = vectorizer.transform(all_api).toarray()
    columns = vectorizer.get_feature_names_out()
    vectorized_df = pd.DataFrame(tmp_np, columns=columns, index=index_list)
    return vectorized_df


def get_summaryinfo(f_json):
    summary_list = []
    for key in summary_key_lists:
        summary_list.append(f_json["summary"][key])
    # print(summary_list)
    return summary_list




def main():
    ##  初期設定変数 ##
    target_dataset = "../custom_datasets/dataset_6/*json"
    max_features = 100
    n_gram = 3

    saving_time_path =f"../experiment/dataset6/vectorizer_time_ver2/tfidf/100/{n_gram}gram/time.txt"

    all_API_list = [] #TF-IDFモデルに渡すリスト
    index_list = []
    all_summary_list = []


    ## データセットのファイルパスを取得 ##
    file_paths = glob.glob(target_dataset)
    # print(f"データセットの数 = {len(file_paths)}")

    ## APIをTF-IDFモデルに渡す形にする ##
    all_API_list, index_list, all_summary_list = get_API_Index_from_file(file_paths)
    

    ## TF-IDFモデルに学習させる #
    start_time = time.time()
    vectorized_df = train_tfidf(all_API_list, n_gram=n_gram, max_features=max_features, index_list=index_list)
    end_time = time.time()

    ## ベクトル化時間を記録 ##
    with open(saving_time_path, mode="w", encoding="utf-8") as timefile:
        print(f"ベクトル化時間 = {end_time - start_time}", file=timefile)

    ## Summary情報を付与する ##
    summary_df = pd.DataFrame(all_summary_list, index=index_list, columns=summary_key_lists)
    result_df = pd.concat((vectorized_df, summary_df), axis=1)

    ## LABELを付ける ##
    result_df["LABEL"] = result_df.index.to_series().apply(lambda x: 0 if 'benreports' in x else 1)


    result_df.to_csv(f"../CSV/dataset6ver2CSV/tfidf/{max_features}/{n_gram}gram/result.csv")
    



if __name__ == "__main__":
    print("----------TF-IDFによるベクトル化----------")
    main()