"""
バケット配列を用いてAPIをベクトル化する。
"""
import pandas as pd
import numpy as np
import time


#【パターン１】 #
def simple_buket_changer(target_str:str) -> float:
    changed_vector = 0.0
    target_len = len(target_str)
    num_buket = 64
    buket_list = [0]*num_buket

    #単純に除算の余りをインデックスに+1
    for char in target_str:
        buket_index = ord(char) % num_buket
        buket_list[buket_index] += 1
    
    return sum((i*value) for i, value in enumerate(buket_list)) / target_len


#【パターン２】#
def nextto_buket_changer(target_str:str) -> float:
    changed_vector = 0.0
    target_len = len(target_str)
    num_buket = 64
    buket_list = [0]*num_buket

    #単純に除算の余りのインデックスに＋１
    for char in target_str:
        buket_index = ord(char) % num_buket
        buket_list[buket_index] += 1

    # 隣り合う２文字を足した余りをインデックスに+1
    for i in range(target_len-1):
        char1, char2 = target_str[i], target_str[i+1]
        buket_index = (ord(char1)+ord(char2)) % num_buket
        buket_list[buket_index] += 1
    
    changed_vector = sum((i*value) for i, value in enumerate(buket_list)) / target_len
    return changed_vector

#【パターン３】#
def Str_position_buket_changer(target_str:str) -> float:
    changed_vector = 0           
    target_len = len(target_str) #文字列の長さ
    num_buket = 64
    buket_list = [0]*num_buket

    #単純に除算の余りのインデックスに＋１
    for char in target_str:
        buket_index = ord(char) % num_buket
        buket_list[buket_index] += 1
    
    # 隣り合う２文字を足した余りをインデックスに+1
    for i in range(target_len-1):
        char1, char2 = target_str[i], target_str[i+1]
        buket_index = (ord(char1)+ord(char2)) % num_buket
        buket_list[buket_index] += 1

    #文字列のi番目も考慮したものを加算
    for i, char in enumerate(target_str):
        buket_index = (ord(char) + i) % num_buket
        buket_list[buket_index] += (i+1)

    
    changed_vector = sum((i*value) for i, value in enumerate(buket_list)) / target_len
    return changed_vector

#【パターン４】#
def Scale_Str_position_buket_changer(target_str:str) -> float:
    changed_vector = 0           
    target_len = len(target_str) #文字列の長さ
    num_buket = 64
    buket_list = [0]*num_buket

    #単純に除算の余りのインデックスに＋１
    for char in target_str:
        buket_index = ord(char) % num_buket
        buket_list[buket_index] += 1
    
    # 隣り合う２文字を足した余りをインデックスに+1
    for i in range(target_len-1):
        char1, char2 = target_str[i], target_str[i+1]
        buket_index = (ord(char1)+ord(char2)) % num_buket
        buket_list[buket_index] += 1

    #文字列のi番目も考慮したものを加算
    for i, char in enumerate(target_str):
        buket_index = (ord(char) + i) % num_buket
        buket_list[buket_index] += (i+1)

    
    changed_vector = sum((0.01*i*value) for i, value in enumerate(buket_list)) / target_len
    return changed_vector

#【パターン５】#
def Simple_and_position_func(target_str:str, num_buket=10, scale_value=1) -> float:
    changed_vector = 0           
    target_len = len(target_str) #文字列の長さ
    buket_list = [0]*num_buket

    #単純に除算の余りのインデックスに＋１
    for char in target_str:
        buket_index = ord(char) % num_buket
        buket_list[buket_index] += 1

    #文字列のi番目も考慮したものを加算
    for i, char in enumerate(target_str):
        buket_index = (ord(char) + i) % num_buket
        buket_list[buket_index] += (0.1*i+1)
    
    changed_vector = sum((scale_value*i*value) for i, value in enumerate(buket_list)) / target_len
    return changed_vector


def main():
    print("\n###############START###############")

    ### 対象のCSVを取得 ###
    csv_path = "../CSV/dataset4CSV/origin/2label_WithoutSummary.csv"
    df = pd.read_csv(csv_path, index_col=0)
    print("df.shape = {}".format(df.shape))

    ###ベクトル化 ###
    start_time = time.time()

    ## 【パターン１】除算だけを扱うベクトル化 ##
    # vectorized_data = df.iloc[:, 0:100].applymap(simple_buket_changer)
    # end_time = time.time()

    ## 【パターン２】除算と隣り合う文字列も考慮したベクトル化 ##
    # vectorized_data = df.iloc[:, 0:100].applymap(nextto_buket_changer)
    # end_time = time.time()

    ##【パターン３】上記に加え、文字列の位置を重みとして加算してベクトル化け##
    # vectorized_data = df.iloc[:, 0:100].applymap(Str_position_buket_changer)
    # end_time = time.time()

    ##【パターン４】パターン３のスケールしたベクトル化 ##
    # vectorized_data = df.iloc[:, 0:100].applymap(Scale_Str_position_buket_changer)
    # end_time = time.time()

    ##【パターン５】パターン１とパターン3の重みを小数にしたとき ##
    vectorized_data = df.iloc[:, 0:100].applymap(Simple_and_position_func)
    end_time = time.time()



    df.iloc[:, 0:100] = vectorized_data

    #CSV化
    df.to_csv("../CSV/dataset4CSV/buket/SimpleAndPositin_2label_WithoutSummary.csv")
    print("Vectorization time is {}s".format(end_time - start_time))
    




    print("#################END#################\n")

if __name__ == "__main__":
    main()