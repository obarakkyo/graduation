"""
ben_reportsデータセットを分割する.

先頭API100個 + Summaryの内容９個


対象外ファイル183個でした。
"""
import json
import glob
import os
import time

def create_summary_features(behavior_json):
    summary_dict = {"command_line"  : 0, 
                    "connects_host" : 0,
                    "connects_ip"   : 0,
                    "directory_created" : 0,
                    "directory_enumerated" : 0,
                    "directory_removed" : 0,
                    "dll_loaded" : 0,
                    "downloads_file" : 0,
                    "fetches_url" : 0,
                    "file_copied" : 0,
                    "file_created" : 0,
                    "file_deleted" : 0,
                    "file_exists" : 0,
                    "file_failed" : 0,
                    "file_moved" : 0,
                    "file_opened" : 0,
                    "file_read" : 0,
                    "file_recreated" : 0,
                    "file_written" : 0,
                    "guid" : 0,
                    "mutex" : 0,
                    "regkey_deleted" : 0,
                    "regkey_opened" : 0,
                    "regkey_read" : 0,
                    "regkey_written" : 0,
                    "resolves_host" : 0,
                    "tls_master" : 0,
                    "wmi_query" : 0,
                    } 
    summary_json = behavior_json["summary"]
    for _, key in enumerate(summary_dict):
        target_key = summary_json.get(key)
        if target_key is not None:
            summary_dict[key] = len(summary_json[key])
    # print(summary_dict)
    return summary_dict
    

def create_api_features(behavior_json, maxnum):
    api_list = []
    count = 0

    process_json = behavior_json["processes"]
    for i in range(len(process_json)):
        calls_json = process_json[i]["calls"]
        if len(calls_json) == 0:
            continue
        else:
            call_num = len(calls_json)
            for j in range(call_num):
                if count == maxnum:
                    break
                api_list.append(calls_json[j]["api"])
                count+=1
    # print(len(api_list))
    api_list = {"api_list":api_list}
    return api_list

def create_filename(file_path, count):
    filename = file_path.replace("../ben_reports\\", "")
    filename = "report" + str(count) + "-" + filename
    # print(filename)
    return filename


def main():
    """初期値変数"""
    target_folder = "../ben_reports/*json"

    failed_file_path = []

    count = 0

    output_count = 0

    """ファイルのパスを取得する。"""
    file_paths = glob.glob(target_folder)
    print("file_paths = {}".format(len(file_paths)))


    """JSONファイルの作成"""
    for file_path in file_paths[:]:
        output_count += 1
        # print("This file is {}".format(file_path))
        try:
            with open(file_path, "r") as f:
                f_json = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decordin JSON from file: {file_path}")
            failed_file_path.append(file_path)
            continue
    
        all_dict = {}

        #behaviorがないなら飛ばす
        if f_json.get("behavior") is None:
            print("This file does not contain [behavior] ")
            failed_file_path.append(file_path)
            continue

        behavior_json = f_json["behavior"]

        #processesがないなら飛ばす
        if behavior_json.get("processes") is None:
            print("This file does not contain [processes] ")
            failed_file_path.append(file_path)
            continue

        #summaryがないなら飛ばす
        if behavior_json.get("summary") is None:
            print("This file does not contain [summary]")
            failed_file_path.append(file_path)
            continue

        ## API抜き出し
        api_dict = create_api_features(behavior_json, maxnum=100)

        ##Summaryの抜出
        summary_dict = create_summary_features(behavior_json)

        ##all_dictに結合
        all_dict = {**api_dict, **summary_dict}

        #保存用ファイル名作成
        count += 1
        filename = create_filename(file_path, count)

        #保存用ファイルパス作成
        newfile_path = "../custom_datasets/obara_bendataset/" + filename

        #ファイルとして保存
        with open(newfile_path, "w", encoding="utf-8") as f2:
            json.dump(all_dict, f2)

        
        #あと何個か出力
        os.system("cls")
        print("{} / {}".format(output_count, len(file_paths)))
        time.sleep(0.1)
    
    for failed_path in failed_file_path:
        print(failed_path)
    print("対象外ファイル数 = {}".format(len(failed_file_path)))


            
    

if __name__ == "__main__":
    print("このプログラムは、ben_reportsデータセットを分割します。")
    main()