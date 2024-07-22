"""
ben_reportsのJSONファイルを分割する。
"""

"""
このプログラムは、ben_reportsデータセットを分割する。

len(file_paths) = 1890
Error decoding JSON from file: ../ben_reports\a088170d91a58e42bda88d192cd50536.json
Error decoding JSON from file: ../ben_reports\db88cea04959ef0e922c90b53738f37a.json
"""
import json
import glob

def create_filepath(file_path, count):
    filename = file_path.replace("../ben_reports\\", "")
    filename = "report" + count + "--" + filename
    filename = "../ben_reports_splits/" + filename
    # print(filename)
    return filename


def create_api_features(behavior_json, maxnum=100):
    get_list = []
    for _, thread_id in enumerate(behavior_json["apistats"]):
        api_json = behavior_json["apistats"][thread_id]
        for _, apiname in enumerate(api_json):
            for i in range(api_json[apiname]):
                if len(get_list) == maxnum:
                    break
                get_list.append(apiname)
    api_dict = {"apis":get_list}
    return api_dict

def create_summary_features(behavior_json):
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
    print("このプログラムは、ben_reportsデータセットを分割する。\n")

    # 対象データセットのパス取得#
    target_folder = "../ben_reports/*json"
    file_paths = glob.glob(target_folder)
    print("len(file_paths) = {}".format(len(file_paths)))

    count = 1

    for file_path in file_paths[:]:
            try:
                with open(file_path, "r") as f:
                    f_json = json.load(f)
            except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {file_path}")

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

            ## 保存用のファイルパスを作成 ##
            filename = create_filepath(file_path, str(count))
            count += 1

            ## API(MAXNUM)個取得
            api_dict = create_api_features(behavior_json, maxnum=100)

            #summaryをぬき出す
            summary_dict = create_summary_features(behavior_json)

            #all_dictに結合
            all_dict = {**api_dict, **summary_dict}

            #ファイルとして保存
            file_save_func(filename, all_dict)





if __name__ == "__main__":
    main()