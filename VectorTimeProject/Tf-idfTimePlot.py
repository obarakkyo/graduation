"""
Base Folder = graduation
"""
import glob
import json
import pandas as pd
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def MyTfidf_Vectorizer(target_paths, n_gram):
    api_list = []

    for path in target_paths:
        with open(path, mode="r", encoding="utf-8") as f:
            file_json = json.load(f)
            tmp_list = []
            tmp_list = file_json["all_api"]
            api_list.append(" ".join(tmp_list))
    
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(n_gram, n_gram))

    start_time = time.time()
    vectorizer.fit(api_list)
    tmp_np = vectorizer.transform(api_list).toarray()
    end_time = time.time()

    columns = vectorizer.get_feature_names_out()

    processing_time = end_time - start_time
    return tmp_np, processing_time, columns


def TfidfTimeMain(status_dict:dict) -> None:
    """
    メイン関数。
    """

    ### データセットのパスを取得 ###
    file_paths = glob.glob(status_dict["target_folder"])

    ## APIが100個に満たないファイルを取りのぞく###
    file_paths.remove("../特徴量エンジニアリング用/customdataset/FFRI2017_report\ML.Attribute.HighConfidence-28e05b9d8a1af2ac620f84f526963f642d11cb78137a9877402678f775c1e749.json")
    print(f"対象データセット = {status_dict['target_folder']}")
    print(f"全データセット数 = {len(file_paths)}")


    ### ベクトル化対象データセットを選択し，ベクトル化関数に渡す###
    with open(status_dict["TimeReportPath"], mode="w", encoding="utf-8") as report:
        for current_gram in [1, 2, 3]:
            for max_search in status_dict["max_search_list"]:
                vectorized_list, report_time, columns = MyTfidf_Vectorizer(file_paths[0:max_search], n_gram=current_gram)
                print(f"[{current_gram}gram]検体数 = {max_search}, ベクトル化時間 = {report_time}", file=report)

                # CSVとしてとりあえず保存 #
                df = pd.DataFrame(vectorized_list, index=file_paths[0:max_search], columns=columns)
                df.to_csv(f"VectorTimeProject/tmp_CSV/[TFIDF][{current_gram}gram]maxsearch{max_search}.csv")

    




if __name__ == "__main__":
    status_dict = {
        "target_folder" : "../特徴量エンジニアリング用/customdataset/FFRI2017_report/*json",
        "max_search_list" : [100, 500, 1000, 2000, 3000, 4000, 5000, 6000],
        # "max_search_list" : [100],
        "TimeReportPath" : "experiment/VectorTimeProjectReport/TfidfTimeRepoer.txt"
    }
    TfidfTimeMain(status_dict)