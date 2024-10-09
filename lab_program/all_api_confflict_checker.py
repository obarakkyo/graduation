"""
all_apilist.csvを利用してベクトル変換の
コンフリクトの割合を調査する。
"""
import pandas as pd
from collections import Counter




def position_buket(api_list:list, weight=0.1, length=20) -> dict:
    buket_list = [0] * length
    vector_dict = {}

    for api_name in api_list:
        for i, char in enumerate(api_name):
            buket_index = ord(char) % length
            buket_list[buket_index] += (weight*i + 1)
        vector =  sum((i*value) for i, value in enumerate(buket_list)) / len(api_name)
        vector_dict[api_name] = vector
    return vector_dict




def simple_buket(api_list:list, length:int=20) -> dict:
    vector_dict = {}
    
    for target_str in api_list:
        buket_list = [0]*length
        for char in target_str:
            buket_index = ord(char) % length
            buket_list[buket_index] += 1
        vector = sum((i*value) for i, value in enumerate(buket_list)) / len(target_str)
        vector_dict[target_str] = vector
    return vector_dict





def confflict_checker(vector_dict:dict) -> dict:
    connfflict_api={}
    connfflict_count = 0
    connfflict_value_list = []

    values = list(vector_dict.values())
    values_counts = Counter(values)

    #コンフリクトしてるvalueをlistに追加#
    for key, value in values_counts.items():
        if value != 1:
            connfflict_count += value
            if key not in connfflict_value_list:
                connfflict_value_list.append(key)

    # コンフリクトしているAPIを抜き出す.#
    for key, value in vector_dict.items():
        if value in connfflict_value_list:
            if value not in connfflict_api:
                connfflict_api[value] = []
            connfflict_api[value].append(key)
    
    #コンフリクトの割合を算出
    confflict_rate = connfflict_count / len(vector_dict) * 100
    # print(confflict_rate)

    return connfflict_api, confflict_rate, connfflict_count




def calculate_ascii_vector(target_str: str) -> float:
    if type(target_str) is int:
        target_str = str(target_str)
    squares = sum(ord(char) ** 2 for char in target_str)
    return squares / len(target_str)

def ascii_func(api_list:list) -> dict:
    vector_dict = {}
    for i in range(len(api_list)):
        target_str = api_list[i]
        vector = calculate_ascii_vector(target_str)
        vector_dict[target_str] = vector
    return vector_dict

def main():
    # all_apilist.csvを読み込む #
    target_csv ="all_apilist.csv" 
    api_df = pd.read_csv(target_csv)
    api_list = []
    for i in range(api_df.shape[0]):
        api_list.append(api_df.iloc[i, 0])

    """ASCIIコードを用いたベクトル化 """
    # vector_dict = ascii_func(api_list)
    # for key, value in vector_dict.items():
    #     print(f"{key:<35}: {value:>15.8f}")
    """------------------------------"""

    """バケット配列とASCII変換を用いたベクトル化"""
    # vector_dict = simple_buket(api_list, length=50)
    # for key, value in vector_dict.items():
    #     print(f"{key:<35}: {value:>15.8f}")
    """----------------------------------------"""

    """バケット配列でi番目の値を考慮したベクトル化"""
    # vector_dict = position_buket(api_list, weight=0.1, length=3)
    # for key, value in vector_dict.items():
    #     print(f"{key:<35}: {value:>15.8f}")
    """------------------------------------------"""






    
    ### コンフリクトをチェックする ###
    # report_dict, confflict_rate, confflict_count = confflict_checker(vector_dict)


    ### 配列の大きさでどのように遷移するか調べる ###
    for i in range(3, 128, 2):
        vector_dict = position_buket(api_list, weight=0.1, length=i)
        # vector_dict = simple_buket(api_list, length=i)
        report_dict, confflict_rate, confflict_count = confflict_checker(vector_dict)
        print("i={}, rate={:.2f}%, count={}".format(i, confflict_rate, confflict_count))

if __name__ == "__main__":
    main()