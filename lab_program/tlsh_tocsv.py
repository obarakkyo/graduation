"""###プログラムの説明###

TLSHを用いた特徴量を作成し、CSVとして保存する。
    ・72文字のハッシュ値を4文字×18個に分解している。

対象データセット：custom_dataset / dataset_1
 Backdoor.Graybird 235
 Packed.Generic    235
 Ransom.Cerber     235
 report            235


"""

"""必要なモジュール"""
import pandas as pd
import glob
import time
import tlsh
import typing

"""TLSHによるハッシュ値を作成し、CSVとして出力する"""
def create_tlsh_tocsv(file_paths: list[str], csv_path: str) -> None:

    start_time = time.time() #<- 開始時間

    spilit_hash_list = []
    
    for i in range(len(file_paths)):
        with open(file_paths[i] ,'rb') as f:
            target_hash = tlsh.hash(f.read())
            inside_spilit_list = []
            for i in range(0, 72, 4):
                inside_spilit_list.append(target_hash[i:i+4])
            spilit_hash_list.append(inside_spilit_list)
    

    ### インデックス名を作成する ###
    index_name = []
    for file_name in file_paths[:]:
        file_name = file_name.replace('custom_datasets/dataset_1\\', '')
        file_name = file_name.replace('.json.txt.txt', '')
        index_name.append(file_name)

    ### データフレーム化 """
    df = pd.DataFrame(spilit_hash_list, index=index_name)


    ### ラベル付け ###
    """
    Backdoor.Graybird 1
    Packed.Generic    2
    Ransom.Cerber     3
    report            0
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
        else:
            print("ラベル付けできないファイルがあります。")
            exit()


    # print(df)

    ### csvとして出力 ###
    df.to_csv(csv_path)

    end_time = time.time()
    print('実行時間 = ', end_time - start_time)
            



"""対象とするデータセットフォルダからファイルパスを取得"""
def get_files_path(folder_path: str) -> list[str]:
    file_paths = glob.glob(folder_path)
    print('データセット   =', folder_path)
    print('対象ファイル数 = ', len(file_paths))
    return file_paths
    



if __name__ == "__main__":
    ### 対象データセットの設定 ###
    folder_path = 'custom_datasets/dataset_1/*txt'

    ### データセット内のファイルのパスを取得 ###
    file_paths = get_files_path(folder_path)

    ### TLSHハッシュ値を計算し、CSVに変換 ###
    csv_path = "CSV/anything/tlsh_csv_origin_4label.csv"
    create_tlsh_tocsv(file_paths, csv_path)
    
    
