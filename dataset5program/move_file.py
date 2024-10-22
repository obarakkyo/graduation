"""
対象とするファイルをコピーして
dataset5に移動させるプログラム。

ただしAIPが100個未満のデータは対象外とする。

"""

import glob
import json
import shutil


# obara_bendataset用のファイルコピー関数 #
def check_bendataset_api_100(file_path:str):
    try:
        with open(file_path, mode="r") as f:
            f_json = json.load(f)
            if len(f_json["api_list"]) < 100:
                return False
            else:
                new_path = file_path.replace("../custom_datasets/obara_bendataset\\", "../custom_datasets/dataset5/")
                shutil.copy(file_path, new_path)
                return True
                
 

    except:
        error_comment = "[ERROR]Can't oepn the file -> " + file_path
        print(error_comment)
        return False

def main():
    """obara_bendatasetから移動させる。"""
    # target_folder = "../custom_datasets/obara_bendataset/*json"
    # file_paths = glob.glob(target_folder)
    # print("\nターゲットフォルダ = {}, 数 = {}".format(target_folder, len(file_paths)))

    # failed_file_path = []
    # move_count = 0


    # # APIが100個以上あるフォルダを移動する #
    # for file_path in file_paths[:]:
    #     if check_bendataset_api_100(file_path):
    #         move_count += 1
    #     else:
    #         failed_file_path.append(file_path)
    

    """obara_virus2017_processから移動させる。"""
    target_virus_name = [
        "Backdoor.Graybird",
        "Packed.Generic",
        "Ransom.Cerber",
        "inforstealer.Limitail",
        "Trojan.Gen",
    ]

    # target_folder = "../custom_datasets/obara_virus2017_process/*json"
    # file_paths = glob.glob(target_folder)
    # print("\nターゲットフォルダ = {}, 数 = {}".format(target_folder, len(file_paths)))

    # for virus_name in target_virus_name:
    #     virus_paths = 
    #     target_virus_paths = glob.glob()
    




if __name__ == "__main__":
    main()