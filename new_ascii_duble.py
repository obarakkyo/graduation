#必要なモジュールのインポート
import pandas as pd
import glob
import time

def main(folder_path, 
         max_columns,
         name_replace,
         CSV_path):
    start_time = time.time()#<--計測開始
    
    """対象フォルダのファイルパス取得"""
    file_paths = glob.glob(folder_path)
    print('Number of all files = ', len(file_paths))
    
    """ベクトル化（2つのAPI関数を対象とする）"""
    csv_list = [] #<--csvにするときに使うリスト
    continue_file = [] #<--max_colmunsを満たさなかったファイルリスト
    for file_path in file_paths[:]:
        with open(file_path, 'r') as f:
            file_tolist = f.read().split(' ')
            API_list = []
            
            #もしmax_columnsを満たさなかったら飛ばす
            if len(file_tolist) < max_columns+1:
                print('file_name =', file_path)
                continue_file.append(file_path)
                continue
            
            for i in range(max_columns):
                square1 = 0
                square2 = 0
                for word1 in file_tolist[i]:
                    square1 += ord(word1) ** 2
                vector1 = square1 / len(file_tolist[i])
                for word2 in file_tolist[i+1]:
                    square2 += ord(word2) ** 2
                vector2 = square2 / len(file_tolist[i+1])
                vector = (vector1 + vector2) / 2
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
            file_name = file_name.replace(name_replace, '')
            index_name.append(file_name)
    new_df = pd.DataFrame(csv_list, index=index_name)
    """
    Nonevirus : 0
    Backdoor  : 1
    Ransom    : 2
    Trojan    : 3
    """
    for index_name in new_df.index[:]:
        if 'report' in index_name:
            new_df.loc[index_name, 'LABEL'] = 0
        else:
            new_df.loc[index_name, 'LABEL'] = 1
    new_df.to_csv(CSV_path)
    
        
    
    
    
    end_time = time.time()#<--計測終了
    result=end_time-start_time
    return result
    
if __name__ == '__main__':
    folder_path = './report8_dataset/*txt'
    max_columns = 100 #API関数を先頭から何個をアスキーコード化するか？ +1
    name_replace = './report8_dataset\\'
    CSV_path = './CSV/Binarization/ascii/ascii_' + str(max_columns) + '_double.csv'
    
    
    result = main(folder_path, max_columns, name_replace, CSV_path)
    print('time:', result)