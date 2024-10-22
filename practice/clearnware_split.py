"""
奈良田さんが収集したクリーンウェアログを分割するプログラム。

apistasがあるプログラム = 726個
apistatsもsummaryもあるプログラム = 713個
"""
import json
import glob

def apis_check(folder_paths):
    count = 0
    for folder_path in folder_paths[:]:
        file_path = folder_path.replace("\\", "/")
        file_path = file_path + "/reports/report.json"
        with open(file_path, "r") as f:
            f_json = json.load(f)

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

            count += 1
    
    print("count = {}".format(count))


def create_api_features(behavior_json, maxnum=100):
    get_list = []
    for _, thread_id in enumerate(behavior_json["apistats"]):
        api_json = behavior_json["apistats"][thread_id]
        for _, apiname in enumerate(api_json):
            for i in range(api_json[apiname]):
                if len(get_list) == 100:
                    break
                get_list.append(apiname)
    api_dict = {"apis":get_list}
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
    file_path = "../custom_datasets/obara_clearn/" + filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_dict, f)
            

def main():
    print("This is main program.")

    target_folder = "../clearn_log/analyses/*"
    print("targetfloder = {}".format(target_folder))

    folder_paths = glob.glob(target_folder)
    print("len(floder_paths) = {}\n".format(len(folder_paths)))

    #behavior->apistats&summaryがあるか確認する。
    # apis_check(folder_paths)

    for folder_path in folder_paths[:]:
        file_path = folder_path.replace("\\", "/")
        file_path = file_path + "/reports/report.json"
        with open(file_path, "r") as f:
            f_json = json.load(f)
            all_dict = {}

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

            #apiを抜き出す
            apis_dict = create_api_features(behavior_json, maxnum=100)
            # print(apis_dict)
            # print(len(apis_dict["apis"]))


            #summaryをむき出す
            summary_dict = create_summary_features(behavior_json)
            # print(summary_dict)

            #all_dictに結合
            all_dict = {**apis_dict, **summary_dict}
            # print(all_dict)

            file_name = file_path.replace("../clearn_log/analyses/", "")
            file_name = file_name.replace("/reports/report", "")
            file_name = "report" + file_name
            print(file_name)

            #ファイルとして保存
            file_save_func(file_name, all_dict)
                



if __name__ == "__main__":
    main()