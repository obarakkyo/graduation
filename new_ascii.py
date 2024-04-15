"""必要なモジュールのインポート"""
import pandas as pd
import glob
import time


def main(folder_path, max_columns, name_replace, CSV_path):
    start_time = time.time() #<--計測時間
    
    """対象ファルダのファイルパス取得"""
    file_paths = glob.glob(folder_path)
    print('全ファイルの数 = ', len(file_paths))
    
    
    
    """ベクトル化"""
    csv_list = [] #<--csvにするときに使うリスト
    continue_file = [] #<--max_columnsに満たさなかったファイル群
    for file_path in file_paths[:]:
        with open(file_path, 'r') as f:
            file_tolist = f.read().split(' ')
            API_list = []
            
            #もしmax_columnsよりAPI関数が少なかったら飛ばす。
            if len(file_tolist) < max_columns:
                print('file_name', file_path)
                continue_file.append(file_path)
                continue
            
            for text in file_tolist[:max_columns]:
                squares = 0
                for word in text:
                    squares += ord(word) ** 2
                vector = squares / len(text)
                API_list.append(vector)
            csv_list.append(API_list)
  
    #確認
    print('対象ファルダの中のファイルの数 = ', len(file_paths))
    print('ベクトル化したファイルの数 = ', len(csv_list))
    
    
    
    """csvとして保存する"""
    index_name = []
    for file_name in file_paths[:]: 
        if file_name in continue_file:
            continue
        else:
            file_name = file_name.replace(name_replace,  '')
            index_name.append(file_name)
    
    new_df  = pd.DataFrame(csv_list, index= index_name)
    """
    Nonevirus : 0
    Backdoor  : 1
    Ransom    : 2
    Trojan    : 3
    """
    # for index_name in new_df.index[:]:
    #     if 'report' in index_name:
    #         new_df.loc[index_name, 'LABEL'] = 0
    #     elif 'Backdoor' in index_name:
    #         new_df.loc[index_name, 'LABEL'] = 1
    #     elif 'Ransom' in index_name:
    #         new_df.loc[index_name, 'LABEL'] = 2
    #     else:
    #         new_df.loc[index_name, 'LABEL'] = 3
    
    """
    Nonevirus : 0
    それ以外   : 1
    """
    for index_name in new_df.index[:]:
        if 'Packed' not in index_name:
            new_df.loc[index_name, 'LABEL'] = 1
        else:
            new_df.loc[index_name, 'LABEL'] = 0
    new_df.to_csv(CSV_path)
    
    
    
    

    end_time = time.time()
    result = end_time - start_time #<--計測終了
    return result
    
if __name__ == "__main__":
    
    folder_path = './report8_dataset/*txt'
    max_columns = 100 #<--ファイルの先頭から何個をアスキーコード変換するか？
    name_replace = './report8_dataset\\' #データフレームのインデックス名用
    CSV_path = './CSV/Binarization/grad/ascii/ascii_' + str(max_columns) + 'report8_Packed.csv'
    
    result = main(folder_path, max_columns, name_replace, CSV_path)
    
    print('経過時間 = ', result)