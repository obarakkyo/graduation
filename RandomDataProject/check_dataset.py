"""
RandomDataProject用のプログラム.

特徴エンジニアリングフォルダにあるクリーンウェアとマルウェアのファイルをすべて
customdataset/RandomDataset内に移動する.

その過程でAPIが100個に満たない検体ファイルや破損ファイルは省く。

ベースディレクトリ = gradustion


[CLEAN]
・ben_reports (全1877)
  移動したファイルの数 = 1261

・cleanware_reports(全1122)
　移動したファイルの数 = 601

・FFRI2017(全6253)
　移動したファイルの数 = 6252

"""
import json
import glob
import shutil #ファイル移動用


def main_move_file():
    # クリーンウェア ben_reports用#
    # targetr_folder = "../特徴量エンジニアリング用/customdataset/benreports_report/*"
    # file_paths = glob.glob(targetr_folder)
    # print(f"ターゲットフォルダ = {targetr_folder}")
    # print(f"ファイル数 = {len(file_paths)}")

    # with open("RandomDataProject/LOG/benreport_removepath.txt", mode="w", encoding="utf-8") as logfile:
    #     for path in file_paths:
    #         with open(path, mode="r", encoding="utf-8") as f:
    #             f_dict = json.load(f)
    #             if len(f_dict["all_api"]) >= 100:
    #                 shutil.copy(path, "custom_datasets/RandomDataset/clean")
    #             else:
    #                 print(f"[LOG]Remove:{path}", file=logfile)
    
    # クリーンウェア ben_reports用#
    # targetr_folder = "../特徴量エンジニアリング用/customdataset/cleanware_report/*"
    # file_paths = glob.glob(targetr_folder)
    # print(f"ターゲットフォルダ = {targetr_folder}")
    # print(f"ファイル数 = {len(file_paths)}")

    # with open("RandomDataProject/LOG/cleanware_report_removepath.txt", mode="w", encoding="utf-8") as logfile:
    #     for path in file_paths:
    #         with open(path, mode="r", encoding="utf-8") as f:
    #             f_dict = json.load(f)
    #             if len(f_dict["all_api"]) >= 100:
    #                 shutil.copy(path, "custom_datasets/RandomDataset/clean")
    #             else:
    #                 print(f"[LOG]Remove:{path}", file=logfile)

    # マルウェア FFRI2017FFRI2017用#
    targetr_folder = "../特徴量エンジニアリング用/customdataset/FFRI2017_report/*"
    file_paths = glob.glob(targetr_folder)
    print(f"ターゲットフォルダ = {targetr_folder}")
    print(f"ファイル数 = {len(file_paths)}")

    with open("RandomDataProject/LOG/malware_report_removepath.txt", mode="w", encoding="utf-8") as logfile:
        for path in file_paths:
            with open(path, mode="r", encoding="utf-8") as f:
                f_dict = json.load(f)
                if len(f_dict["all_api"]) >= 100:
                    shutil.copy(path, "custom_datasets/RandomDataset/malware")
                else:
                    print(f"[LOG]Remove:{path}", file=logfile)




if __name__ == "__main__":
    main_move_file()