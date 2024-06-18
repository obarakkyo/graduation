"""
TLSHのハッシュ値CSVを取得して、提案手法でベクトル化する。

"""
import glob
import pandas as pd
import time
import math

###ベクトル化関数###
def calculate_ascii_vector(target_str: str) -> float:
    squares = sum(ord(char) ** 2 for char in target_str)
    mean_squares = squares / len(target_str)

    #スケーリングファクター使用
    scaling_factor = 1000.0
    scaled_mean_squares = mean_squares / scaling_factor

    sigmoid_value = 1 / (1 + math.exp(-scaled_mean_squares))
    return sigmoid_value

def main() -> None:
    print('This is main program!')

    ###実行前に指定する変数###
    target_csv    = "../CSV/dataset1CSV/origin/tlsh_csv_origin_4spilit_4label.csv"
    csv_save_path = "../CSV/dataset1CSV/ascii/tlsh_AsciiSigmoid_4split_2label.csv"

    ####CSVの取得####
    df = pd.read_csv(target_csv, index_col=0)
    print(df.head())
    print("df.shape = {}".format(df.shape))

    ###ベクトル化###
    vectorized_data = df.iloc[:, :-1].applymap(calculate_ascii_vector)
    

    ###CSV化###
    new_df = pd.DataFrame(vectorized_data, index=df.iloc[:].index)
    print(new_df.head())
    print(new_df.shape)

    """
    Nonevirus : 0
    それ以外   : 1
    """
    ###ラベル付け###
    new_df['LABEL'] = df.index.map(lambda index_name: 0 if 'report' in index_name else 1)

    new_df.to_csv(csv_save_path)

            

            

if __name__ == "__main__":
    print("This program changes TLSH to ASCII Code!")
    main()