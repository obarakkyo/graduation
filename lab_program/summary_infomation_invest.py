"""
virusLog2017, ClearnLog, Ben_reportsを対象に
Summary情報のキーをすべて取得する。
"""

import glob
import json
import csv


def jsonfile_check_func(file_paths:list, SummaryKey_list:list, failed_file_path:list):
    for file_path in file_paths[:]:
        #ファイルが開けるかチェック
        try:
            with open(file_path, mode="r") as f:
                target_json = json.load(f)
        except:
            print("[ERROR]このファイルは開けませんでした。")
            error_comment = "[Can't open]" + file_path
            failed_file_path.append(error_comment)
            continue

        #behaviorがあるかチェック
        if target_json.get("behavior") is None:
            print("[ERROR]このファイルにはbehaviorがありません。")
            error_comment = "[no behavior]" + file_path
            failed_file_path.append(error_comment)
            continue
        else:
            behavior_json = target_json["behavior"]
        
        #summaryがあるかチェック
        if behavior_json.get("summary") is None:
            print("[ERROR]このファイルにはsummaryがありません。")
            error_comment = "[no summary]" + file_path
            failed_file_path.append(error_comment)
            continue
        else:
            summary_json = behavior_json["summary"]
            SummaryKey_list = summary_check_func(SummaryKey_list, summary_json)

    return SummaryKey_list, failed_file_path
    


def summary_check_func(summary_list:list, summary_json:dict) -> list:
    for key, value in summary_json.items():
        if key not in summary_list:
            summary_list.append(key)
    return summary_list

def main():
    print("\nこのプログラムはSummary情報のキーをすべて取得します。")

    SummaryKey_list = []
    failed_file_path = []

    ### VirusLog2017 ###
    target_folder_path = "../FFRI_Dataset_2017/*.json"
    file_paths = glob.glob(target_folder_path)
    print("\ntarget_folder = {}".format(target_folder_path))
    print("All file counts = {}".format(len(file_paths)))

    SummaryKey_list, failed_file_path = jsonfile_check_func(file_paths, SummaryKey_list, failed_file_path)

    
    
    ### ClearnLog ###
    target_folder_path = "../clearn_log/analyses/*"
    folder_paths = glob.glob(target_folder_path)
    print("\ntarget_folder = {}".format(target_folder_path))
    print("All file counts = {}".format(len(folder_paths)))
    file_paths = []
    for folder_path in folder_paths[:]:
        tmp_path = folder_path + "/reports/report.json"
        tmp_path = tmp_path.replace("\\", "/")
        file_paths.append(tmp_path)
    
    SummaryKey_list, failed_file_path = jsonfile_check_func(file_paths, SummaryKey_list, failed_file_path)



    ### Ben_reports ###
    target_folder_path = "../ben_reports/*.json"
    file_paths = glob.glob(target_folder_path)
    print("\ntarget_folder = {}".format(target_folder_path))
    print("All file counts = {}".format(len(file_paths)))
    
    SummaryKey_list, failed_file_path = jsonfile_check_func(file_paths, SummaryKey_list, failed_file_path)
    
    ### 最終確認 ###
    SummaryKey_list = sorted(SummaryKey_list)
    print("len(SummaryKey_list) = ", len(SummaryKey_list))
    
    ### CSVとして出力 ###
    with open("All_Summary_Key.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        for item in SummaryKey_list:
            writer.writerow([item])

    with open("ALL_Summary_Key_ERROR_file.csv", mode="w", newline="") as f1:
        writer = csv.writer(f1)
        for item in failed_file_path:
            writer.writerow([item])







if __name__ == "__main__":
    main()