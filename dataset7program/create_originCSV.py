"""
データセット7のoriginCSVを作成する。
"""
import pandas as pd
import glob
import json
from tqdm import tqdm




# 辞書型のデータを取得し，特徴量リストとして返す#
def create_features_func(file_dict):
    return_list = []
    api_list = []
    summary_list = []

    #APIを抽出
    api_list = file_dict["all_api"]
    api_list = api_list[0:100]

    #Summaryと親子情報を抽出
    summary_dict = file_dict["summary"]
    for _, value in summary_dict.items():
        summary_list.append(value)
    
    summary_list.append(file_dict["parent_count"])
    summary_list.append(file_dict["children_count"])

    return_list = api_list + summary_list
    return return_list


def main():
    ### データセットのパス ###
    dataset_path = "../custom_datasets/dataset_7/*.json"
    saving_csv_path = "../CSV/dataset7CSV/origin/2label.csv"


    ### 格納するリストを宣言 ###
    index_list = []
    features_list = []
    index_name = ""



    ### ファイルのパスをリストに取得 ##
    file_paths = glob.glob(dataset_path)
    print(f"ターゲットデータセット = {dataset_path}")
    print(f"ファイルの数 = {len(file_paths)}")


    ###リストに値を格納していく ###
    for file_path in tqdm(file_paths[:]):
        with open(file_path, mode="r", encoding="utf-8") as f:
            file_dict = json.load(f) #辞書型に変換

            # 特徴量を作成#
            features_list.append(create_features_func(file_dict))

            #インデックス名を作成#
            index_name = file_path.replace("../custom_datasets/dataset_7\\", "")
            index_name = index_name.replace(".json", "")
            index_list.append(index_name)
    
    # 列名を作成 #
    columns_list = [f"API{i+1}" for i in range(100)]
    for _, value in file_dict["summary"].items():
        columns_list.append(value)
    columns_list.append("parent")
    columns_list.append("children")

    # データフレームに変換 #
    df = pd.DataFrame(features_list, index=index_list, columns=columns_list)

    # LABEL付け#
    df["LABEL"] = df.index.to_series().apply(lambda x: 0 if "benreport" in str(x) else 1)

    # 保存 # 
    df.to_csv(saving_csv_path)
            

            


    

if __name__ == "__main__":
    main()
