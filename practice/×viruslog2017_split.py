"""

virusLog2017のJSONファイルから
・API100個

"""

import glob
import json
# import pickle

def create_filename(virusname, file_path):
    if virusname == None:
        file_name = "NULL"
    else:
        file_name = virusname
    file_path = file_path.replace('../FFRI_Dataset_2017\\', '-')
    # file_path = file_path.replace(".json", ".txt")
    file_name += file_path
    # print("filename = {}".format(file_name))
    return file_name

def create_features(f_json, maxnum_api=100):
    get_list = [0 for i in range(maxnum_api)]
    idx = 0
    for _, thread_id in enumerate(f_json["behavior"]["apistats"]):
        api_json = f_json["behavior"]["apistats"][thread_id]
        for _, api_name in enumerate(api_json):
            for i in range(api_json[api_name]):
                if idx == maxnum_api:
                    break
                get_list[idx] = api_name
                idx += 1
                # print(idx, get_list[idx-1])
    api_dict = {"apis":get_list}
    return api_dict

def file_save_func(filename, all_dict):
    file_path = "../custom_datasets/obara_virus2017/" + filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_dict, f)

def create_summary_features(f_json):
    summary_dict = {"file_created"   : 0,
                    "file_recreated" : 0,
                    "file_deleted"   : 0,
                    "file_written"   : 0,
                    "file_opened"    : 0,
                    "regkey_written" : 0,
                    "regkey_opened"  : 0,
                    "regkey_read"    : 0,
                    "command_line"   : 0
                    } 
    summary_json = f_json["behavior"]["summary"]
    
    for idx, key in enumerate(summary_dict):
        target_key = summary_json.get(key)
        if target_key is not None:
            summary_dict[key] = len(summary_json[key])

    return summary_dict
     




def main():
    print('==========main==========')

    target_folder = "../FFRI_Dataset_2017/*.json"
    print(f"target_folder = {target_folder}")

    file_paths = glob.glob(target_folder)
    print(f"Total number of files = {len(file_paths)}")

    
    for file_path in file_paths[:]:
        print("This file is {}".format(file_path))
        with open(file_path, "r") as f:
            f_json = json.load(f)
            all_dict = {}

            #symantecがあるか確認
            scans_json = f_json["virustotal"]["scans"]
            if scans_json.get("Symantec") is None:
                continue

            #保存ファイル名作成
            virus_name = f_json["virustotal"]["scans"]["Symantec"]["result"]
            filename = create_filename(virus_name, file_path)

            #APIの抜き出し
            apis_dict = create_features(f_json, maxnum_api=100)

            #summaryから抜き出し
            summary_dict = create_summary_features(f_json)

            #all_dictに結合
            all_dict = {**apis_dict, **summary_dict}
            # print(all_dict)

            #ファイルとして保存
            file_save_func(filename, all_dict)

        # print('----------')

if __name__ == "__main__":
    print('This program create original txt!')
    main()