"""###プログラムの説明###

TLSHを用いた特徴量を作成し、CSVとして保存する。
    ・72文字のハッシュ値を4文字×18個に分解している。




"""

"""必要なモジュール"""
import pandas as pd
import glob
import time
import tlsh
import typing

"""TLSHによるハッシュ値を作成し、リストにして返す"""
#４つごとに分解する関数
def create_tlsh_4_split_list(file_paths: list[str]) -> list[str]:
    spilit_hash_list = []
    
    for i in range(len(file_paths)):
        with open(file_paths[i] ,'rb') as f:
            target_hash = tlsh.hash(f.read())
            inside_spilit_list = []
            for i in range(0, 72, 4):
                inside_spilit_list.append(target_hash[i:i+4])
            spilit_hash_list.append(inside_spilit_list)
    return spilit_hash_list

#３つごとに分解する関数
def create_tlsh_3_split_list(file_paths: list[str]) -> list[str]:
    spilit_hash_list = []
    
    for i in range(len(file_paths)):
        with open(file_paths[i] ,'rb') as f:
            target_hash = tlsh.hash(f.read())
            inside_spilit_list = []
            for i in range(0, 72, 3):
                inside_spilit_list.append(target_hash[i:i+3])
            spilit_hash_list.append(inside_spilit_list)
    return spilit_hash_list

#2つごとに分解する関数
def create_tlsh_2_split_list(file_paths: list[str]) -> list[str]:
    spilit_hash_list = []
    for i in range(len(file_paths)):
        with open(file_paths[i], 'rb') as f:
            target_hash = tlsh.hash(f.read())
            inside_split_list = []
            for j in range(0, 72, 2):
                inside_split_list.append(target_hash[j:j+2])
            spilit_hash_list.append(inside_split_list)
    return spilit_hash_list

#１つごとに分解する関数
def create_tlsh_1_split_list(file_paths: list[str]) -> list[str]:
    spilit_hash_list = []
    for i in range(len(file_paths)):
        with open(file_paths[i], 'rb') as f:
            target_hash = tlsh.hash(f.read())
            inside_split_list = []
            for j in range(72):
                inside_split_list.append(target_hash[j])
            spilit_hash_list.append(inside_split_list)
    return spilit_hash_list

     

"""インデックス名を作成""" 
def create_index_name_list(file_paths):
    index_name = []
    for file_name in file_paths[:]:
        file_name = file_name.replace('custom_datasets/dataset_2\\', '')
        file_name = file_name.replace('.json.txt.txt', '')
        index_name.append(file_name)
    return index_name


"""ラベル付けをする関数"""
def labeling_func(df) -> pd.DataFrame:
    """
    Backdoor.Graybird         1
    Packed.Generic            2
    Ransom.Cerber             3
    Infostealer.Limitail      4
    Trojan.Gen                5
    report                    0
    """
    for index_name in df.index[:]:
        if index_name.startswith("report"):
            df.loc[index_name, 'LABEL'] = 0
        elif index_name.startswith("Backdoor"):
            df.loc[index_name, 'LABEL'] = 1
        elif index_name.startswith("Packed"):
            df.loc[index_name, 'LABEL'] = 2
        elif index_name.startswith("Ransom"):
            df.loc[index_name, 'LABEL'] = 3
        elif index_name.startswith("Infostealer.Limitail"):
            df.loc[index_name, 'LABEL'] = 4
        elif index_name.startswith("Trojan.Gen"):
            df.loc[index_name, 'LABEL'] = 5
        else:
            print("ラベル付けできないファイルがあります。")
            print("index_name = ", index_name)
            exit()
    
    return df

    # print('実行時間 = ', end_time - start_time)
            



"""対象とするデータセットフォルダからファイルパスを取得"""
def get_files_path(folder_path: str) -> list[str]:
    file_paths = glob.glob(folder_path)
    print('データセット   =', folder_path)
    print('対象ファイル数 = ', len(file_paths))
    return file_paths
    



if __name__ == "__main__":
    ### 対象データセットの設定 ###
    folder_path = 'custom_datasets/dataset_2/*txt'

    ### データセット内のファイルのパスを取得 ###
    file_paths = get_files_path(folder_path)

    ### TLSHハッシュ値を計算し、CSVに変換 ###
    csv_path = "CSV/dataset2CSV/origin/tlsh_csv_origin_1_spilit_6label.csv"

    # spilit_hash_list = create_tlsh_4_split_list(file_paths)
    # spilit_hash_list = create_tlsh_3_split_list(file_paths)
    # spilit_hash_list = create_tlsh_2_split_list(file_paths)
    spilit_hash_list = create_tlsh_1_split_list(file_paths)

    index_name_list = create_index_name_list(file_paths)
    print(index_name_list)

    ###データフレーム化###
    df = pd.DataFrame(spilit_hash_list, index=index_name_list) 

    # print(df)

    ###ラベル付け###
    df = labeling_func(df)

    ### csvとして出力 ###
    df.to_csv(csv_path)


     
    
