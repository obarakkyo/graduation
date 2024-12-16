"""
データセット7を対象に，TF-IDFによるベクトル化を行うプログラム．

"""
import json
import glob
import time
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

# APIリストでTF-IDFを学習させる #
def train_tfidf(all_api:list, n_gram=1, max_features=100, index_list = []):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(n_gram, n_gram))
    vectorizer.fit(all_api)

    tmp_np = vectorizer.transform(all_api).toarray()
    columns = vectorizer.get_feature_names_out()
    vectorized_df = pd.DataFrame(tmp_np, columns=columns, index=index_list)
    return vectorized_df




def get_API_Index_from_file(_file_paths:list):
    api_list = []
    index_list = []
    all_summary_list = []

    for file_path in _file_paths[:]:
        tmp_string = ""
        tmp_summary=[]
        index_list.append(file_path.replace("../../custom_datasets/dataset_7\\", ""))
        with open(file_path, mode="r") as f:
            f_json = json.load(f)
            for _, value in enumerate(f_json["all_api"]):
                tmp_string += str(value) + " "
            api_list.append(tmp_string)

            tmp_summary = get_summaryinfo(f_json)
            all_summary_list.append(tmp_summary)
    return api_list, index_list, all_summary_list


def get_summaryinfo(f_json):
    summary_list = []
    for key in summary_key_lists:
        summary_list.append(f_json["summary"][key])
    summary_list.append(f_json["parent_count"])
    summary_list.append(f_json["children_count"])
    return summary_list


def tfidf_main():
    #　対象ファイルパスの取得 #
    target_folder = "../../custom_datasets/dataset_7/*json"
    file_paths = glob.glob(target_folder)
    print(f"ファイルの数 = {len(file_paths)}")

    # 初期変数 #
    all_API_list = [] #TF-IDFモデルに渡すリスト
    index_list = []
    all_summary_list = []

    # フォルダのAPIをすべて抽出する #
    all_API_list, index_list, all_summary_list = get_API_Index_from_file(file_paths)

    # 学習用のループ #
    n_gram = [1, 2, 3]
    max_features = 100

    for current_gram in n_gram:
        saving_time_path = f"../../experiment/dataset7/vectorizer_time/tfidf/{current_gram}gram_time.txt"

        start_time = time.time()
        vectorized_df = train_tfidf(all_API_list, n_gram=current_gram, max_features=max_features, index_list=index_list)
        end_time = time.time()

        ## ベクトル化時間を記録 ##
        with open(saving_time_path, mode="w", encoding="utf-8") as timefile:
            print(f"ベクトル化時間 = {end_time - start_time}", file=timefile)

        ## Summary情報を付与する ##
        summary_parent_children_columns = summary_key_lists + ["parent", "children"]
        summary_df = pd.DataFrame(all_summary_list, index=index_list, columns=summary_parent_children_columns)
        result_df = pd.concat((vectorized_df, summary_df), axis=1)

        ## LABELを付ける ##
        result_df["LABEL"] = result_df.index.to_series().apply(lambda x: 0 if 'benreports' in x else 1)


        result_df.to_csv(f"../../CSV/dataset7CSV/tfidf/max{max_features}_{current_gram}gram.csv")

        print(f"終了：{current_gram}gram.")





if __name__ == "__main__":
    tfidf_main()