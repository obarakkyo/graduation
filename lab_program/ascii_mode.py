"""
アスキー変換の後に除算の余りを算出して
ハッシュ値（数値）を計算するプログラム。
"""
import glob
import pandas as pd

def main():
    print('This is ascii_mod program!\n')

    ### 実行前に指定する変数 ###
    target_folder = "../custom_datasets/dataset_2/*"
    csv_save_path = "../CSV/dataset2CSV/ascii_mod/ascii_mod_2label.csv"

    ### 対象フォルダ内のファイル名を取得
    filenames = glob.glob(target_folder)
    print("All files = {}".format(len(filenames)))

if __name__ == "__main__":
    main()