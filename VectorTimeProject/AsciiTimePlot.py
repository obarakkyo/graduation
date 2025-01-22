import glob
import json
import time
import numpy as np
import pandas as pd


def base_ascii_vector(target_str, LogBool=False):
    """
    ASCIISquareのベースプログラム
    引数：文字列
    戻り値：ベクトル化した値
    """
    if type(target_str) is int:
        target_str = str(target_str)
    squares = sum(ord(char) ** 2 for char in target_str)

    if LogBool:
        #対数変換して返す
        return np.log1p(squares / len(target_str))
    else: 
        return squares / len(target_str)
    



def ascii_vectorizer(target_paths:list):
    """
    ベクトル化対象ファイルパスを受け取り，ベクトル化を仲介する関数．
    """
    all_API_List = []
    tmp_API_List = []

    ## 前処理 ##
    for path in target_paths:
        with open(path, mode="r", encoding="utf-8") as f:
            file_dict = json.load(f)
            tmp_API_List = file_dict["all_api"]
            all_API_List.append(tmp_API_List[0:100])
    
    start_time = time.time() #計測開始
    for i in range(len(all_API_List)):
        for j in range(100):
            try :
                all_API_List[i][j] = base_ascii_vector(all_API_List[i][j], LogBool=False)
            except:
                print(f"[LOG]API数が足りないので飛ばします．{target_paths[i]}")
                continue
    end_time = time.time() #計測終了

    processing_time = end_time - start_time
    return all_API_List, processing_time
    

def ascii_main(status_dict:dict) -> None:
    """
    メインプログラム。
    AsciiSquare手法による，ベクトル化時間を調査。
    """
    
    ### データセットからパスを取得###
    file_paths = glob.glob(status_dict["target_folder"])

    ## APIが100個に満たないファイルを取りのぞく###
    file_paths.remove("../特徴量エンジニアリング用/customdataset/FFRI2017_report\ML.Attribute.HighConfidence-28e05b9d8a1af2ac620f84f526963f642d11cb78137a9877402678f775c1e749.json")
    print(f"対象データセット = {status_dict['target_folder']}")
    print(f"全データセット数 = {len(file_paths)}")


    ### ベクトル化対象範囲のリストを選択 ###
    with open(status_dict["TimeReportPath"], mode="w", encoding="utf-8") as report:
        for max_search in status_dict["max_search_list"]:
            vectorized_list , report_time = ascii_vectorizer(file_paths[max_search:max_search+1000])
            print(f"検体数 = {max_search}~{max_search+1000}, ベクトル化時間 = {report_time}", file=report)

            # CSVとしてとりあえず保存 #
            df = pd.DataFrame(vectorized_list, index=file_paths[max_search:max_search+1000])
            df.to_csv(f"VectorTimeProject/tmp_CSV/[ASCII]maxsearch{max_search}_{max_search+1000}.csv")

if __name__ == "__main__":
    status_dict = {
        "target_folder" : "../特徴量エンジニアリング用/customdataset/FFRI2017_report/*json",
        "max_search_list" : [0, 1000, 2000, 3000, 4000, 5000],
        "TimeReportPath" : "experiment/VectorTimeProjectReport/AsciiTimeRepoer.txt"
    }
    ascii_main(status_dict)