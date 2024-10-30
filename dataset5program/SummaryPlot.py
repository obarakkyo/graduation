"""
dataset5を参考にSummary情報をPlotするプログラムの作成
"""

import matplotlib.pyplot as plt
import glob 
import json
import os
import statistics
import pandas as pd

summary_key_lists = ["command_line", 
                    "connects_host",
                    "connects_ip",
                    "directory_created",
                    "directory_enumerated",
                    "directory_removed",
                    "dll_loaded",
                    "downloads_file",
                    "fetches_url",
                    "file_copied",
                    "file_created",
                    "file_deleted",
                    "file_exists",
                    "file_failed",
                    "file_moved",
                    "file_opened",
                    "file_read",
                    "file_recreated",
                    "file_written",
                    "guid",
                    "mutex",
                    "regkey_deleted",
                    "regkey_opened",
                    "regkey_read",
                    "regkey_written",
                    "resolves_host",
                    "tls_master",
                    "wmi_query"]



def get_summary_tolist(summary_key:str, file_paths) -> list:
    get_num_list = []
    count = 0
    for path in file_paths[:]:
        with open(path, mode="r") as f:
            f_json = json.load(f)
            summary_num = f_json[summary_key]
            get_num_list.append(summary_num)
            count += 1
            # if count % 100 == 0:
            #     print("{} / {}".format(count, len(file_paths)))
    return get_num_list

def plot_hist_func(get_list:list, summary_key:str, file_class:str, get_min=0, get_max=10) -> None:
     # 取得したリストを利用してヒストグラムを表示 #
    # plt.hist(get_list, bins=[i for i in range(100)], color="blue")
    plt.hist(get_list, bins=[i for i in range(get_max+1)],  color="blue")
    plt.title(f"{summary_key}'s Histgram [{file_class}]")
    # plt.xlim(0, get_max)
    plt.ylim(0, 1300)
    plt.xlabel("num")
    plt.ylabel("Frequency")
    # plt.show()

    save_path = "../experiment/dataset5/SummartPlotPicture/" + summary_key + "_" + file_class + ".png"
    plt.savefig(save_path)


def main():

    # データフレームにして後でCSVで保存すつためのリスト #
    static_list = []
    tmp_list = [] #max, min, mea, median
    name_list = [] #index_name

    ## Summaryのキー情報を取得する ##
    for summary_key in summary_key_lists[:]:
        print("##### SummaryKey is {} #####".format(summary_key))


        """クリーンウェアのパターン"""
        target_folder = "../custom_datasets/dataset5/report*"
        file_paths = glob.glob(target_folder)
        print("Target folder = {}, len = {}".format(target_folder, len(file_paths)))

        # 値をリストで取得する # 
        get_normal_list = get_summary_tolist(summary_key, file_paths)
        tmp_max = max(get_normal_list)
        tmp_min = min(get_normal_list)
        tmp_mean = statistics.mean(get_normal_list)
        tmp_median = statistics.median(get_normal_list)
        normal_variance = statistics.pstdev(get_normal_list)

        print("max    = {}".format(tmp_max))
        print("min    = {}".format(tmp_min))
        print("mean   = {}".format(tmp_mean))
        print("median = {}".format(tmp_median))
        print("variance = {}".format(normal_variance))
        tmp_list = [tmp_max, tmp_min, tmp_mean, tmp_median, normal_variance]
        static_list.append(tmp_list)
        index_name = "[Normal]" + summary_key
        name_list.append(index_name)

        """マルウェアのパターン"""
        target_folder = "../custom_datasets/dataset5/*"
        file_paths = [f for f in glob.glob(target_folder) if not os.path.basename(f).startswith('report')]
        print("Target folder = {}, len = {}".format(target_folder, len(file_paths)))

        # 値をリストで取得する # 
        get_malware_list = get_summary_tolist(summary_key, file_paths)
        tmp_max = max(get_malware_list)
        tmp_min = min(get_malware_list)
        tmp_mean = statistics.mean(get_malware_list)
        tmp_median = statistics.median(get_malware_list)
        malware_variance = statistics.pstdev(get_malware_list)
        print("max    = {}".format(tmp_max))
        print("min    = {}".format(tmp_min))
        print("mean   = {}".format(tmp_mean))
        print("median = {}".format(tmp_median))
        print("variance = {}".format(malware_variance))

        tmp_list = [tmp_max, tmp_min, tmp_mean, tmp_median, malware_variance]
        static_list.append(tmp_list)
        index_name = "[Malware]" + summary_key
        name_list.append(index_name)



        if normal_variance > malware_variance:
            my_max = int(normal_variance * 3)
        else:
            my_max = int(malware_variance * 3)

        if my_max <= 20:
            my_max = 20
        elif my_max > 100:
            my_max = 200

        # if summary_key in  ["directory_enumerated", "file_exists"]:
        #     continue

        # プロットする #
        # plot_hist_func(get_normal_list, summary_key, file_class="Normal", get_max=my_max)

        # プロットする #
        # plot_hist_func(get_malware_list, summary_key, file_class="Malware", get_max=my_max)

    # CSVとして保存#
    df = pd.DataFrame(static_list, index=name_list, columns=["max", "min", "mean", "median", "pstev"])
    df.to_csv("../experiment/dataset5/SummartPlotPicture/static.csv")




if __name__ == "__main__":
    print("このプログラムはSummary情報をプロットします.")
    main()