"""
TLSHのハッシュ値CSVを取得して、提案手法でベクトル化する。

"""
import glob
import pandas as pd
import time

###ベクトル化関数###
def calculate_ascii_vector(target_str: str) -> float:
    if type(target_str) is int:
        target_str = str(target_str)
    squares = sum(ord(char) ** 2 for char in target_str)
    return squares / len(target_str)

def calculate_ascii_vector_1split(target_str: str) -> int:
    return ord(str(target_str))

def create_label(df):
    pass


def main() -> None:
    print('This is main program!')

    ###実行前に指定する変数###
    target_csv    = "../CSV/dataset2CSV/origin/tlsh_csv_origin_2_spilit_6label.csv"
    csv_save_path = "../CSV/dataset2CSV/ascii/tlsh_ascii_2split_2label.csv"

    ####CSVの取得####
    df = pd.read_csv(target_csv, index_col=0)
    print(df.head())
    print("df.shape = {}".format(df.shape))

    ###ベクトル化###
    vectorized_data = df.iloc[:, :-1].applymap(calculate_ascii_vector)
    # vectorized_data = df.iloc[:, :-1].applymap(calculate_ascii_vector_1split)

    

    ###CSV化###
    new_df = pd.DataFrame(vectorized_data, index=df.iloc[:].index)
    print(new_df.head())
    print(new_df.shape)


    ###ラベル付け###

    #２値の場合 #
    new_df['LABEL'] = df.index.map(lambda index_name: 0 if 'report' in index_name else 1)

    # N値の場合 #
    # for index_name in new_df.index[:]:
    #     if index_name.startswith("report"):
    #         new_df.loc[index_name, 'LABEL'] = 0
    #     elif index_name.startswith("Backdoor"):
    #         new_df.loc[index_name, 'LABEL'] = 1
    #     elif index_name.startswith("Packed"):
    #         new_df.loc[index_name, 'LABEL'] = 2
    #     elif index_name.startswith("Ransom"):
    #         new_df.loc[index_name, 'LABEL'] = 3
    #     elif index_name.startswith("Infostealer.Limitail"):
    #         new_df.loc[index_name, 'LABEL'] = 4
    #     elif index_name.startswith("Trojan.Gen"):
    #         new_df.loc[index_name, 'LABEL'] = 5
    #     else:
    #         print("ラベル付けできないファイルがあります。")
    #         print("index_name = ", index_name)
    #         exit()


    new_df.to_csv(csv_save_path)

            

            

if __name__ == "__main__":
    print("This program changes TLSH to ASCII Code!")
    main()