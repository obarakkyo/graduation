"""
バケット配列の数を増やしたときに、
コンフリクトしているAPIの数の推移を調べる。
"""
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def calculate_ascii_vector(target_str: str) -> float:
    if type(target_str) is int:
        target_str = str(target_str)
    squares = sum(ord(char) ** 2 for char in target_str)
    return squares / len(target_str)

def buket_change_func(target_str:str, scale_num=1, num_buket=64):
    changed_vector = 0           
    target_len = len(target_str) #文字列の長さ
    buket_list = [0]*num_buket

    #単純に除算の余りのインデックスに＋１
    for char in target_str:
        buket_index = ord(char) % num_buket
        buket_list[buket_index] += 1
    
    # # 隣り合う２文字を足した余りをインデックスに+1
    # for i in range(target_len-1):
    #     char1, char2 = target_str[i], target_str[i+1]
    #     buket_index = (ord(char1)+ord(char2)) % num_buket
    #     buket_list[buket_index] += 1

    # # #文字列のi番目も考慮したものを加算
    for i, char in enumerate(target_str):
        buket_index = (ord(char)) % num_buket
        buket_list[buket_index] += (0.1*i+1)

    # # #2乗の除算をインデックスに+1
    # for char in target_str:
    #     buket_index = (ord(char)**2) % num_buket
    #     buket_list[buket_index] += 1

    # #XORを利用したベクトル化
    # for i in range(target_len - 1):
    #     char1, char2 = target_str[i], target_str[i + 1]
    #     buket_index = (ord(char1) ^ ord(char2)) % num_buket  # XORを使用してバケットインデックスを決定
    #     buket_list[buket_index] += 1

    
    changed_vector = sum((scale_num*i*value) for i, value in enumerate(buket_list)) / target_len
    return changed_vector

def main():
    print("\n#########START##########")

    ###初期設定###
    csv_path = "../CSV/dataset4CSV/origin/2label_WithoutSummary.csv"

    ###データの読み込み###
    df = pd.read_csv(csv_path, index_col=0)
    print("df.shape = ", df.shape)

    ###特徴量の抽出###
    feature_data = df.iloc[:, 0:-1]
    print("feature_data.shape = ", feature_data.shape)
    # print(feature_data.head())


    ###全部のAPIをリストにする。###
    all_api_list = []
    api_dict = {}
    for i in range(0,df.shape[0]):
        for j in range(0, 100):
            target_api = feature_data.iloc[i, j]
            # print(target_api)
            if target_api not in all_api_list:
                all_api_list.append(target_api)


    ###コンフリクトの推移を確かめる###
    confflict_dict = {}
    confflict_count_list = []

    for i in range(10, 150, 5):
        ###ベクトル化###
        for api in all_api_list:
            vector = buket_change_func(api, scale_num=1, num_buket=i) #buket
            # vector = calculate_ascii_vector(api) #ASCII
            api_dict[api] = vector
            # print("{} = {}".format(api, vector))
        
        ###昇順にソート###
        sorted_api_dict = dict(sorted(api_dict.items(), key=lambda item:item[1], reverse=False))

        ###衝突している値を見つける###
        values = list(sorted_api_dict.values())
        value_counts = Counter(values)
        confflict_count = 0
        confflict_value = []
        for key, value in value_counts.items():
            if value != 1:
                confflict_count += value
                confflict_value.append(key)
        print("[{}]Confflict = {}, / {}".format(i, confflict_count, len(values)))
        confflict_count_list.append(confflict_count)
        confflict_dict[i] = confflict_count

        for key, value in api_dict.items():
            if value in confflict_value:
                print("API=[{}], VALUE=[{}]".format(key, value))
        print()
    
    print(confflict_dict)
    print(confflict_count_list)

    

if __name__ == "__main__":
    main()