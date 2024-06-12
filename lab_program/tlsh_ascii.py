"""
TLSHのハッシュ値CSVを取得して、提案手法でベクトル化する。

"""
import glob
import pandas as pd
import time

def main() -> None:
    print('This is main program!')

    ####CSVの取得####
    target_csv = "../CSV/dataset1CSV/origin/tlsh_csv_origin_4spilit_4label.csv"
    df = pd.read_csv(target_csv, index_col=0)
    print(df.head())
    print("df.shape = {}".format(df.shape))

    ###ベクトル化###
    csv_list = []
    # for i in range(df.shape[0]):
    #     for j in range(df.shape[1]-1):

    ###お試し###
    df_1 = df.iloc[0:5, :]
    print(df_1.shape)
    print(df_1.shape[0])
    print(len(df_1))
    
    for i in range(len(df_1)):
        inside_ascii_list = []
        for j in range(df_1.shape[1] -1 ):
            target_str = df_1.iloc[i, j]
            squares = 0.0
            vector  = 0.0
            for t in range(len(target_str)):
                squares += ord(target_str[t]) ** 2
            vector = squares / len(target_str)
            inside_ascii_list.append(vector)
        csv_list.append(inside_ascii_list)

    for i in range(len(csv_list)):
        print(csv_list[i])
    

    ###CSV化###
    new_df = pd.DataFrame(csv_list, index=df.iloc[0:5, :].index)
    print(new_df.head())
    print(new_df.shape)

    """
    Nonevirus : 0
    それ以外   : 1
    """
    for index_name in new_df.index[:]:
        if 'report' not in index_name:
            new_df.loc[index_name, 'LABEL'] = 1
        else:
            new_df.loc[index_name, 'LABEL'] = 0
    new_df.to_csv("../CSV/dataset1CSV/ascii/test_ascii.csv")

            

            

if __name__ == "__main__":
    print("This program changes TLSH to ASCII Code!")
    main()