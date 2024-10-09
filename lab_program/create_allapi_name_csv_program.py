"""
virusLog2017とBen_reports,clearnlog
からそれぞれ存在するすべてのAPI名を抽出し
JSON形式でそれらを出力するプログラム。
"""

import glob
import json
import csv
import pandas




def virusLog2017_to_apilist():
    # 対象フォルダからファイルパスリストを取得#
    target_folder = "../FFRI_Dataset_2017/*json"
    files_paths = glob.glob(target_folder)
    print("target_folder = {}".format(target_folder))
    print("len(target_folder) = {}".format(len(files_paths)))

    api_list = [] #登場するAPIを保持する（重複はなし）
    failed_file_path = []
    
    for i in range(len(files_paths)):
        target_file = files_paths[i]

        #ファイルが開けるか試す#
        try:
            with open(target_file, "r") as f:
                target_json = json.load(f)
        except :
            print("このファイルは破損しています。")
            failed_file_path.append(target_file)
        
        #behaviorがあるか確かめる
        if target_json.get("behavior") is None:
            print("このファイルにはbehaviorがありません。")
            failed_file_path.append(target_file)
            continue
        else:
            behavior_json = target_json["behavior"]
        
        #processesがあるか確かめる
        if behavior_json.get("processes") is None:
            print("このファイルにはprocessesがありません。")
            failed_file_path.append(target_file)
            continue

        
        
        #まだ登場してないAPIが無いかチェックする
        api_list = check_add_apifunc(api_list, behavior_json)

        if i%100 == 0:
            print("{}/{}".format(i, len(files_paths)))
    
    print("\n検索対象から抜いたファイルの数 = {}".format(len(failed_file_path)))

    print("\n抜き出した登場するAPIのリスト = {}".format(len(api_list)))
    api_list = sorted(api_list)
    
    #CSVとして出力
    with open("virusLog2017_apilist.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        for item in api_list:
            writer.writerow([item])
    
    #一応対象外にしたファイルパスも見てみる
    with open("virusLog2017_remove_file.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        for item in failed_file_path:
            writer.writerow([item])




def clearnLog_to_apilist():
    # 対象フォルダからフォルダのパスを取得 #
    target_folder = "../clearn_log/analyses/*"
    folder_paths = glob.glob(target_folder)
    print("target_dataset = {}".format(target_folder))
    print("len(folder_paths) = {}".format(len(folder_paths)))

    api_list = []
    failed_file_path = []

    for i in range(len(folder_paths)):
        folder_paths[i] = folder_paths[i].replace("\\", "/")
        target_file = folder_paths[i] + "/reports/report.json"
        
        #ファイルが開けるか試す
        try:
            with open(target_file, "r") as f:
                target_json = json.load(f)
        except:
            print("このファイルは開けませんでした。")
            error_comment = "[won't open]:" + target_file
            failed_file_path.append(error_comment)
            continue
        
        #behaviorがあるか確かめる
        if target_json.get("behavior") is None:
            print("このファイルにはbehaviorがない")
            error_comment = "[don't behavior]:" + target_file
            failed_file_path.append(error_comment)
            continue
        else:
            behavior_json = target_json["behavior"]
        
        #processesがあるか確かめる
        if behavior_json.get("processes") is None:
            print("このファイルにはprocessesがない")
            error_comment = "[don't processes]:" + target_file
            failed_file_path.append(error_comment)
            continue
        
        #まだ登場していないAPIが無いかチェックする
        api_list = check_add_apifunc(api_list, behavior_json)

        if i%100 == 0:
            print("{}/{}".format(i, len(folder_paths)))
    
    print("\n検索対象から抜いたファイルの数 = {}".format(len(failed_file_path)))
    for file_name in failed_file_path:
        print(file_name)

    print("\n登場するAPIの数 = {}".format(len(api_list)))
    api_list = sorted(api_list)

    # CSVとして出力 #
    with open("clearnLog_apilist.csv", mode="w", newline="") as f1:
        writer = csv.writer(f1)
        for item in api_list:
            writer.writerow([item])

    #対象外にしたファイルもCSVとして保存 #
    with open("clearnLog_remove_file.csv", mode="w", newline="") as f2:
        writer = csv.writer(f2)
        for item in failed_file_path:
            writer.writerow([item])
        

def benreports_to_apilist():
    #対象フォルダからファイルパスを取得
    target_folder = "../ben_reports/*"
    file_paths    = glob.glob(target_folder)
    print("target_folder = {}".format(target_folder))
    print("len(file_paths) = {}".format(len(file_paths))) 

    api_list = []
    failed_file_path = []

    for i in range(len(file_paths)):
        #ファイルが開けるか試す
        try:
            with open(file_paths[i], "r") as f:
                target_json = json.load(f)
        except:
            print("このファイルは開けませんでした")
            error_comment = "[won't open]:" + file_paths[i]
            failed_file_path.append(error_comment)
            continue

        #behaviorがあるか確かめる
        if target_json.get("behavior") is None:
            print("このファイルにはbehaviorがない")
            error_comment = "[don't behavior]:" + file_paths[i]
            failed_file_path.append(error_comment)
            continue
        else:
            behavior_json = target_json["behavior"]
        
        #processes
        if behavior_json.get("processes") is None:
            print("このファイルにはprocessesがない")
            error_comment = "[don't processes]: " + file_paths[i]
            failed_file_path.append(error_comment)
            continue

        #まだ登場していないAPIが無いかチェックする
        api_list = check_add_apifunc(api_list, behavior_json)

        if i%100 == 0:
            print("{}/{}".format(i, len(file_paths)))
    print("\n検索対象から抜いたファイルの数 = {}".format(len(failed_file_path)))
    for file_name in failed_file_path:
        print(file_name)

    print("\n登場するAPIの数 = {}".format(len(api_list)))
    api_list = sorted(api_list)

    # CSVとして出力 #
    with open("benreports_apilist.csv", mode="w", newline="") as f1:
        writer = csv.writer(f1)
        for item in api_list:
            writer.writerow([item])

    #対象外にしたファイルもCSVとして保存 #
    with open("benreports_remove_file.csv", mode="w", newline="") as f2:
        writer = csv.writer(f2)
        for item in failed_file_path:
            writer.writerow([item])

    

def check_add_apifunc(api_list, behavior_json) -> list:
    proccess_json = behavior_json["processes"]
    for i in range(len(proccess_json)):
        calls_json = proccess_json[i]["calls"]
        if len(calls_json) == 0:
            continue
        else:
            call_num = len(calls_json)
            for j in range(call_num):
                if calls_json[j]["api"] not in api_list: #リストにない場合は追加
                    api_list.append(calls_json[j]["api"])
    
    return api_list

def main():
    print("\nこのプログラムはすべてのAPI名を取得したJSONを作ります.")

    """##########VirusLog2017##########"""
    # virusLog2017_to_apilist()
    """################################"""


    """##########ClearnLog##########"""
    # clearnLog_to_apilist()
    """#############################"""


    """##########ben_reports########"""
    benreports_to_apilist()
    """#############################"""


if __name__ == "__main__":
    main()