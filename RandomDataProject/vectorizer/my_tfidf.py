"""
RandomDataProject用のTf-idfベクトル化プログラム
ベースディレクトリ=graduation
"""

import glob
import json
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



def get_summaryinfo(f_json):
    summary_list = []
    for key in summary_key_lists:
        summary_list.append(f_json["summary"][key])
    summary_list.append(f_json["parent_count"])
    summary_list.append(f_json["children_count"])
    return summary_list

def tfidf_main(csv_num):
    # ランダムに選択したマルウェアのパスを取得する #
    malwarepaths = []
    with open(f"RandomDataProject/LOG/result{csv_num}_malwarepath.txt", mode="r", encoding="utf-8") as pathfile:
        for path in pathfile:
            malwarepaths.append(path.replace("\n", "")) #改行を取り除く

    
    All_APIlist = []
    All_indexlist = []
    All_summarylist = []


    # クリーンウエアの情報を抽出する # 
    print("クリーンウェアのデータを抽出する")
    cleanware_paths = glob.glob("custom_datasets/RandomDataset/clean/*")
    for path in cleanware_paths:
        tmp_string = ""
        tmp_summary = []
        with open(path, mode="r", encoding="utf-8") as cleanfile:
            CleanDict = json.load(cleanfile)

            # APIをTFIDF用に連結してリストに追加#
            for _, value in enumerate(CleanDict["all_api"]):
                tmp_string += str(value) + " "
            All_APIlist.append(tmp_string)

            # Summary情報をリストに追加 #
            tmp_summary = get_summaryinfo(CleanDict)
            All_summarylist.append(tmp_summary)

            #インデックス名を取得し格納 #
            All_indexlist.append(path.replace("custom_datasets/RandomDataset/clean\\", ""))
    


    # マルウェアの情報を取得する # 
    print("マルウェアの情報を抽出する")
    for path in malwarepaths:
        tmp_string = ""
        tmp_summary = []
        with open(path, mode="r", encoding="utf-8") as malwarefile:
            MalwareDict = json.load(malwarefile)

            # APIをTFIDF用に連結してリストに追加#
            for _, value in enumerate(MalwareDict["all_api"]):
                tmp_string += str(value) + " "
            All_APIlist.append(tmp_string)

            # Summary情報をリストに追加 #
            tmp_summary = get_summaryinfo(MalwareDict)
            All_summarylist.append(tmp_summary)

            #インデックス名を取得し格納 #
            All_indexlist.append(path.replace("custom_datasets/RandomDataset/malware\\", ""))

    # TF-IDF学習 #
    print("学習を開始")
    n_gram = [1, 2, 3]
    max_features = 100

    for current_num in n_gram:

        start_time = time.time()
        vectorized_df = train_tfidf(all_api=All_APIlist, n_gram=current_num, max_features=max_features, index_list=All_indexlist)
        end_time = time.time()

        # 処理時間を記録 #
        with open(f"experiment/RandomDataProject/TimeLog/tfidf/[{current_num}gram]result{csv_num}.txt", mode="w", encoding="utf-8") as timefile:
            print(f"処理時間 = {end_time - start_time}", file=timefile)

        # Summary情報を付与する #
        summary_parent_children_columns = summary_key_lists + ["parent", "children"]
        summary_df = pd.DataFrame(All_summarylist, index=All_indexlist, columns=summary_parent_children_columns)
        result_df = pd.concat((vectorized_df, summary_df), axis=1)

        # LABELを付ける #
        result_df["LABEL"] = result_df.index.to_series().apply(lambda x: 0 if 'report' in x else 1)

        # CSV保存 #
        result_df.to_csv(f"CSV\RandomDatasetCSV/tfidf/[{current_num}gram]result{csv_num}.csv")
    
    print(f"#####result{csv_num}終了#####")






if __name__ == "__main__":
    for i in range(10):
        tfidf_main(i+1)    
