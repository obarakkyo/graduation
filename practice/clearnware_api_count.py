"""
クリーンウェアの中でAPIが100個あるやつをカウントする。
"""
import glob
import json
import shutil

def check_api(f_json):
    api_list = f_json["api_list"]
    # print(api_list)

    if len(api_list) >= 100:
        return True
    else:
        return False

def main():
    print("This program checks api'scount.")

    target_folder = "../custom_datasets/obara_bendataset/*json"
    print("target_folder = {}".format(target_folder))

    file_paths = glob.glob(target_folder)
    print(file_paths[0])
    print("Total counts of file = {}".format(len(file_paths)))

    #APIが100個あるか確認
    count = 0
    move_files = []
    for file_path in file_paths[:]:
        with open(file_path, "r") as f:
            f_json = json.load(f)
            flag = check_api(f_json)
            if flag:
                count+=1
                # if count > 235:
                #     continue
                move_files.append(file_path)
    print(len(move_files))
    
    #対象ファイルを移動する
    print("Total counts of moving file = {}".format(len(move_files)))
    for file_path in move_files:
        new_path = file_path.replace("../custom_datasets/obara_bendataset\\", "../custom_datasets/dataset_4/")
        shutil.copy2(file_path, new_path)
        
    
    print("count = {}".format(count))
            




if __name__ == "__main__":
    main()