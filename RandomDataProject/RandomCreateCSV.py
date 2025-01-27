"""
ランダムにマルウェアを抽出してCSVを作り、
tmp_CSVフォルダにあるクリーンウェアをくっつけて
CSVファルダに保存する。

クリーンウェアはすべて1862個存在する。

ベースディレクトリ = graduation
"""

import glob
import json
import pandas as pd
import numpy as np
import random #ランダムチョイス用
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


def RandomCreateCSV_main():
    """
    メイン関数 
    """
    #マルウェアのフェーズ# 
    target_folder = "custom_datasets/RandomDataset/malware/*"
    file_paths = glob.glob(target_folder)
    print(f"ターゲットフォルダ = {target_folder}")
    print(f"ファイルの数  = {len(file_paths)}")

    for i in range(10):
        # ランダムに1862個のファイルを抽出 #
        random_paths =random.sample(file_paths, 1862)

        malware_features = []
        malware_index = []

        # 特徴量作成 #
        for path in random_paths[:]:
            with open(path, mode="r", encoding="utf-8") as f:
                file_dict = json.load(f)
                tmp_api = file_dict["all_api"][0:100]
                tmp_summary = get_summary_features(file_dict)
                tmp_features = tmp_api + tmp_summary
            malware_features.append(tmp_features)
            malware_index.append(path.replace("custom_datasets/RandomDataset/malware\\", ""))
        
        # 列名作成#
        columns_list = [f"API{i+1}" for i in range(100)]
        columns_list += summary_key_lists
        columns_list.append("parent")
        columns_list.append("children")

        # データフレームに変換 #
        malware_df = pd.DataFrame(malware_features, index=malware_index, columns=columns_list)
        malware_df["LABEL"] = 1

        # クリーンウェアのCSVを取得
        clean_df = pd.read_csv("RandomDataProject/tmp_CSV/clean.csv", index_col=0)

        # 合体
        result_df = pd.concat([clean_df, malware_df], axis=0)

        # データを保存
        result_df.to_csv(f"CSV/RandomDatasetCSV/origin/result{i+1}.csv")

        # 取得したマルウェアのパスを保存
        with open(f"RandomDataProject/LOG/result{i+1}_malwarepath.txt", mode="w", encoding="utf-8") as logfile:
            for path in random_paths:
                print(path, file=logfile)
    



if __name__ == "__main__":
    RandomCreateCSV_main()