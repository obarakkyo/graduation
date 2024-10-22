"""
virusLog2017のJSONファイルを
API先頭100個＋Summaryから９個分割した
JSONファイルとして分割する。
"""
import json
import glob


def create_filepath(filename):
    file_path = "../custom_datasets/obara_virus2017_process/" + filename
    return file_path

def create_filename(virusname, file_path):
    if virusname == None:
        file_name = "NULL"
    else:
        file_name = virusname
    file_path = file_path.replace('../FFRI_Dataset_2017\\', '-')
    file_name += file_path
    return file_name

def create_features(behavior_json, maxnum=100):
    api_list = []
    count = 0
    ## processesがあるか確認##
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
                count += 1
    # print(api_list, len(api_list))
    api_dict = {"api_list":api_list}
    return api_dict

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
    return summary_dict

def file_save_func(filename, all_dict):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_dict, f)


def main():
    print("このプログラムはVirusLog2017を対象としています。")

    """初期値変数"""
    target_folder = "../FFRI_Dataset_2017/*json"


    """ファイルパスの取得"""
    file_paths = glob.glob(target_folder)
    print("Total count of file = {}".format(len(file_paths)))


    """JSONファイルの作成"""
    for file_path in file_paths[:]:
        # print("This file is {}".format(file_path))
        with open(file_path, "r") as f:
            f_json = json.load(f)
            all_dict = {}

            #symantecがあるか確認
            scans_json = f_json["virustotal"]["scans"]
            if scans_json.get("Symantec") is None:
                continue

            #behaviorが無いなら飛ばす
            if f_json.get("behavior") is None:
                continue

            behavior_json = f_json["behavior"]
            
            #apistsが無いなら飛ばす
            if behavior_json.get("apistats") is  None:
                continue

            #summaryが無いなら飛ばす
            if behavior_json.get("summary") is None:
                continue
            

            #APIの抜出し
            api_dict = create_features(behavior_json, maxnum=100)

            #summaryをぬき出す
            summary_dict = create_summary_features(behavior_json)

            #all_dictに結合
            all_dict = {**api_dict, **summary_dict}

            #保存ファイル名作成
            virus_name = f_json["virustotal"]["scans"]["Symantec"]["result"]
            filename = create_filename(virus_name, file_path)
            # print(filename)

            ## 保存用のファイルパスを作成 ##
            filepath = create_filepath(filename)
            print(filepath)

            #ファイルとして保存
            file_save_func(filepath, all_dict)




if __name__ == "__main__":
    main()