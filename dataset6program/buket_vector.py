"""
バケット配列を用いてAPIをベクトル化する。
"""
import pandas as pd
import numpy as np
import time


# 文字列のi番目も考慮したものを加算 #
def position_buket(target_str:str, scale_num=0.01, num_buket=64):
    changed_vector = 0           
    target_len = len(target_str) #文字列の長さ
    buket_list = [0]*num_buket

    for i, char in enumerate(target_str):
        buket_index = (ord(char)) % num_buket
        buket_list[buket_index] += (0.1*i+1)
    
    changed_vector = sum((scale_num*i*value) for i, value in enumerate(buket_list)) / target_len
    return changed_vector



def main():
    print("\n###############START###############")

    ### 対象のCSVを取得 ###
    csv_path = "../CSV/dataset6CSV/origin/2label.csv"
    df = pd.read_csv(csv_path, index_col=0)
    print("df.shape = {}".format(df.shape))

    ###ベクトル化 ###
    start_time = time.time()
    vectorized_data = df.iloc[:, 0:100].applymap(position_buket)
    end_time = time.time()



    df.iloc[:, 0:100] = vectorized_data

    #CSV化
    df.to_csv("../CSV/dataset6CSV/bucket/Position_64_2label.csv")
    print("Vectorization time is {}s".format(end_time - start_time))
    




    print("#################END#################\n")

if __name__ == "__main__":
    main()