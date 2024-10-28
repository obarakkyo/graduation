"""
dataset6作成のために対象となるデータセットを
特徴量エンジニアリングフォルダから引っ張ってくる。
"""


import glob
from pathlib import Path
import json
import shutil


def main():
    current_path = Path(__file__)
    print(f"current_path = {current_path}")
    target_folder = str(current_path.parent.parent.parent) + "\\特徴量エンジニアリング用\\customdataset\\*"
    print(f"target_folder = {target_folder}")

    folder_paths = glob.glob(target_folder)
    for i, path in enumerate(folder_paths):
        print(f"[{i}] : {path}")
    
    print() #改行
    
    """ FFRIの場合 """
    # target_dataset = folder_paths[2]
    # malware_list = ["Backdoor.Graybird", "infostealer.Limitail", "Packed.Generic", "Ransom.cerber", "Trojan.Gen"]
    # for malware in malware_list:
    #     tmp_path = target_dataset + "\\" + malware + "*"
    #     malware_paths = glob.glob(tmp_path)
    #     print(f"malware'len = {len(malware_paths)}")
    #     for i in range(250):
    #         new_path = malware_paths[i].replace("特徴量エンジニアリング用\customdataset\FFRI2017_report\\","graduation\custom_datasets\dataset_6\\")
    #         shutil.copy(malware_paths[i], new_path)

    """ Ben_reportsの場合"""
    target_folder = folder_paths[0] + "\\*"
    benreports_paths = glob.glob(target_folder)
    print(f"Ben_reports'len = {len(benreports_paths)}")
    failed_count = 0
    
    for current_path in benreports_paths[:]:
        try:
            with open(current_path, mode="r") as f:
                f_json = json.load(f)
                all_api = f_json["all_api"]
                if len(all_api) < 100:
                    error_comment = "[ERROR]API is not enough : " + current_path
                    failed_count += 1
                    continue
                else:
                    print("ここだよ")
                    new_path = current_path.replace("特徴量エンジニアリング用\customdataset\\benreports_report\\","graduation\custom_datasets\dataset_6\\")
                    shutil.copy(current_path, new_path)
 

        except:
            error_comment = "[ERROR]Can't Open : " + current_path
            print(error_comment)
            failed_count += 1
    
    print(f"failed_count = {failed_count}")
    print(f"Moving files = {len(benreports_paths) - failed_count}")




if __name__ == "__main__":
    main()