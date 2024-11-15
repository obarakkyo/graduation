"""
データセット３\4内のJSONファイルからCSVを作成する。
"""
import glob
import pandas as pd
import json

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

def create_feature_list_func(file_path):
    api_lists = []
    summary_lists = []
    parent_children = []
    result = []
    with open(file_path, "r") as f:
        #辞書型にして読み込む
        f_json = json.load(f)

        #API先頭100個を取得
        api_lists = f_json["all_api"]
        api_lists = api_lists[0:100]

        #Summary情報を取得
        summary_dict = f_json["summary"]
        for key in summary_key_lists:
            summary_lists.append(summary_dict[key])

        #親プロセス子プロセスの数を取得
        parent_children.append(f_json["parent_count"])
        parent_children.append(f_json["children_count"])

        result = api_lists + summary_lists + parent_children
        return result

def create_index_name(file_path):
    indexname = file_path.replace("../custom_datasets/dataset_6\\", "")
    indexname = indexname.replace(".json", "")
    # print(indexname)
    return indexname

def create_culmuns():
    culmns_name_list = []
    for i in range(100):
        tmp_name = "API" + str(i+1)
        culmns_name_list.append(tmp_name)
    culmns_name_list += summary_key_lists
    culmns_name_list += ["parent_count", "children_count"]
    return culmns_name_list
    


    

"""ラベル付けをする関数"""
def labeling_func(df) -> pd.DataFrame:
    # """6ラベルVERSION
    # Backdoor.Graybird         1
    # Packed.Generic            2
    # Ransom.Cerber             3
    # Infostealer.Limitail      4
    # Trojan.Gen                5
    # benreport                    0
    # """
    # for index_name in df.index[:]:
    #     if index_name.startswith("benreport"):
    #         df.loc[index_name, 'LABEL'] = 0
    #     elif index_name.startswith("Backdoor"):
    #         df.loc[index_name, 'LABEL'] = 1
    #     elif index_name.startswith("Packed"):
    #         df.loc[index_name, 'LABEL'] = 2
    #     elif index_name.startswith("Ransom"):
    #         df.loc[index_name, 'LABEL'] = 3
    #     elif index_name.startswith("Infostealer.Limitail"):
    #         df.loc[index_name, 'LABEL'] = 4
    #     elif index_name.startswith("Trojan.Gen"):
    #         df.loc[index_name, 'LABEL'] = 5
    #     else:
    #         print("ラベル付けできないファイルがあります。")
    #         print("index_name = ", index_name)
    #         exit()
    # return df

    """2ラベルVERSION
    report    0
    それ以外　1
    """
    for index_name in df.index[:]:
        if index_name.startswith("benreport"):
            df.loc[index_name, 'LABEL'] = 0
        else:
            df.loc[index_name, 'LABEL'] = 1
    return df





def main():
    # 対象フォルダの指定 #
    target_folder_path = "../custom_datasets/dataset_6/*json"
    target_file_paths = glob.glob(target_folder_path)
    print("len(target_file_paths) = {}".format(len(target_file_paths)))

    # データフレーム作成用のリスト宣言#
    all_lists = []
    indexname_lists = []

    #　CSVの保存先宣言 #
    csv_path = "../CSV/dataset6ver2CSV/origin/2label.csv" 

    for file_path in target_file_paths[:]:
        all_lists.append(create_feature_list_func(file_path))
        indexname_lists.append(create_index_name(file_path))
    
    
    # データフレーム化 #
    culmun_list = create_culmuns()

    # print(culmun_list)
    df = pd.DataFrame(all_lists, index=indexname_lists, columns=culmun_list)

    # ラベル付け #
    df = labeling_func(df)
    
    # CSVとして保存#
    df.to_csv(csv_path)

if __name__ == "__main__":
    print("-----This program changes dataset6 to CSV!!-----")
    main()