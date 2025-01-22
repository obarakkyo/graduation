import glob
import json
import time
import pandas as pd

def base_position_bucket(target_str, bucket_len):
    change_vector = 0
    bucket_index = 0

    target_len = len(target_str) #文字列の長さ
    bucket_list = [0]*bucket_len #空のバケット配列

    for i, char in enumerate(target_str):
        bucket_index = ord(char) % bucket_len
        bucket_list[bucket_index] += (0.1*i) + 1
    
    change_vector = sum((i*value) for i , value in enumerate(bucket_list)) / target_len
    # print(change_vector)
    return change_vector




def PositionBucket_main(target_paths:list, bucket_len=128):
    all_API_list = []
    tmp_API_list = []

    ##  前処理 ##
    for path in target_paths:
        with open(path, mode="r", encoding="utf-8") as f:
            file_dict = json.load(f)
            tmp_API_list = file_dict["all_api"]
            all_API_list.append(tmp_API_list[0:100])
    
    start_time = time.time() #計測開始

    for i in range(len(all_API_list)):
        for j in range(100):
            try:
                all_API_list[i][j] = base_position_bucket(all_API_list[i][j], bucket_len)
            except:
                int(f"[LOG]API数が足りないので飛ばします．{target_paths[i]}")
                continue
    end_time   = time.time() #計測終了
    processing_time = end_time - start_time
    return all_API_list, processing_time


def PositionBucketTime(status_dict):
    ### データセットからパスを取得 ###
    file_paths = glob.glob(status_dict["target_folder"])

    # API数が100に満たないパスを削除
    file_paths.remove("../特徴量エンジニアリング用/customdataset/FFRI2017_report\ML.Attribute.HighConfidence-28e05b9d8a1af2ac620f84f526963f642d11cb78137a9877402678f775c1e749.json")
    print(f"対象データセット = {status_dict['target_folder']}")
    print(f"全データセット数 = {len(file_paths)}")



    ###  ベクトル化対象範囲をループ ###
    with open(status_dict["TimeReportPath"], mode="w", encoding="utf-8") as report:
        for bucket_len in status_dict["Bucket_len"]:
            for max_search in status_dict["max_search_list"]:
                vectorized_list, report_time = PositionBucket_main(file_paths[max_search:max_search+1000], bucket_len)
                print(f"検体数 = {max_search}~{max_search+1000}, [len={bucket_len}]ベクトル化時間 = {report_time}", file=report)

                # CSVとしてとりあえず保存 #
                df = pd.DataFrame(vectorized_list, index=file_paths[max_search:max_search+1000])
                df.to_csv(f"VectorTimeProject/tmp_CSV/[Position][{bucket_len}]maxsearch{max_search}_{max_search+1000}.csv")


if __name__ == "__main__":
    status_dict = {
        "target_folder" : "../特徴量エンジニアリング用/customdataset/FFRI2017_report/*json",
        "max_search_list" : [0, 1000, 2000, 3000, 4000, 5000, 6000],
        "Bucket_len" : [64, 128],
        "TimeReportPath" : "experiment/VectorTimeProjectReport/PositionBucketTimeReport.txt"
    }
    PositionBucketTime(status_dict)