"""
・virusLog2017のすべてのAPIリスト
・clearnLogのすべてのAPIリスト
・ben_reportのすべてのAPIリスト
を統合したAPICSVを作る.
"""
import pandas as pd
import csv

def main():
    target_csv = "virusLog2017_apilist.csv"
    target_df = pd.read_csv(target_csv)
    print("virusLog2017 = {}".format(target_df.shape))

    api_list = []

    #virusLog2017.csvのAPIをリストにコピー
    for i in range(target_df.shape[0]):
        api_list.append(target_df.iloc[i, 0])
    print("len(api_list) = {}\n".format(len(api_list)))
    
    #clearnLog.csvのAPIを取得し、リストに無ければ追加する.
    target_csv = "clearnLog_apilist.csv"
    target_df = pd.read_csv(target_csv)
    print("clearnLog = {}".format(target_df.shape))
    for i in range(target_df.shape[0]):
        get_apiname = target_df.iloc[i, 0]
        if get_apiname not in api_list:
            api_list.append(get_apiname)
    print("len(api_list) = {}\n".format(len(api_list)))

    #ben_reports.csvのAPI取得し、リストに無ければ追加する。
    target_csv = "benreports_apilist.csv"
    target_df  = pd.read_csv(target_csv)
    print("bemreports = {}".format(target_df.shape))
    for i in range(target_df.shape[0]):
        get_apiname = target_df.iloc[i, 0]
        if get_apiname not in api_list:
            api_list.append(get_apiname)
    print("len(api_list) = {}\n".format(len(api_list)))

    api_list = sorted(api_list)


    #api_listをCSVとして保存する。
    with open("all_apilist.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        for item in api_list:
            writer.writerow([item])
    
    

if __name__ == "__main__":
    print("\nThis program create all api list.\n")
    main()