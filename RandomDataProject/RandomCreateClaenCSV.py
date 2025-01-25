"""
ランダムにマルウェアを抽出したCSVを作成し保存する.

ベースディレクトリ = graduation
"""

import json
import glob
import pandas as pd
import numpy as np

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


def get_summary_features(data_dict:dict) -> list:
    """
    辞書型のデータを受け取り、Summary情報を抽出して
    リスト型で返す
    """
    result = []

    # Summary情報を追加 #
    for key in summary_key_lists:
        result.append(data_dict["summary"][key])
    
    # 親子プロセス情報を追加 #
    result.append(data_dict["parent_count"])
    result.append(data_dict["children_count"])

    return result






def main_RandomCreateCSV():
    """
    メイン関数
    """

    clean_index = []
    claen_features = []

    #クリーンウェア#
    target_folder = "custom_datasets/RandomDataset/clean/*"
    file_paths = glob.glob(target_folder)
    print(f"ターゲットフォルダ = {target_folder}")
    print(f"ファイルの数 = {len(file_paths)}")

    # 特徴量作成 #
    for path in file_paths[:]:
        with open(path, mode="r", encoding="utf-8") as f:
            file_dict = json.load(f)
            tmp_api = file_dict["all_api"][0:100]         #API100抽出
            tmp_summary = get_summary_features(file_dict) #Summary情報を抽出
            tmp_features = tmp_api + tmp_summary
        claen_features.append(tmp_features)
        clean_index.append(path.replace("custom_datasets/RandomDataset/clean\\", ""))
    
    columns_list = [f"API{i+1}" for i in range(100)]
    columns_list += summary_key_lists
    columns_list.append("parent")
    columns_list.append("children")

    # CSVとして一時保存 #
    claen_df = pd.DataFrame(claen_features, index=clean_index, columns=columns_list)
    claen_df["LABEL"] = 0
    claen_df.to_csv("RandomDataProject/tmp_CSV/claen.CSV")
    
    

        







if __name__ == "__main__":
    main_RandomCreateCSV()