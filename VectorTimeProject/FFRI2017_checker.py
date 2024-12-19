"""
FFRI2017内の情報を調査するためのプログラム。
"""
import glob
import json
from tqdm import tqdm
import time
import pandas as pd

def timer(func):
    def wrapper(*args, **kwards):
        start_time = time.time()
        result = func(*args, **kwards)
        end_time   = time.time()
        print(f"RESULT = {result}")
        print(f"実行時間 = {end_time - start_time}")
        return result
    return wrapper


@timer
def check_APIcount_num(num:int, file_paths=None) -> int:
    """
    APIがnum個以上存在する検体の総数を算出する。
    """
    count = 0 #戻り値（対象ファイルの総数）
    tmp = 0   #一時保存変数

    for i in tqdm(range(len(file_paths))):
        with open(file_paths[i], mode="r", encoding="utf-8") as f:
            file_dict = json.load(f)
            tmp = len(file_dict["all_api"])
            if tmp >= num:
                count += 1
    return count



def checker_main():
    """
    メインプログラム関数。
    """
    #カレントディレクトリは，graduationを想定。
    target_folder = "../特徴量エンジニアリング用/customdataset/FFRI2017_report/*json" #FFRI2017
    # target_folder = "../特徴量エンジニアリング用/customdataset/benreports_report/*json" #benreports

    file_paths = glob.glob(target_folder)
    print(f"データセットの総数 = {len(file_paths)}")



    # APIが100個以上存在するファイル総数を調査#
    print(check_APIcount_num(num=100, file_paths=file_paths))


if __name__ == "__main__":
    checker_main()