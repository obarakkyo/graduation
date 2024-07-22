#必要なモジュールのインポート
import pandas as pd
import numpy as np
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
import time

def main(
    folder_name,
    n_gram,
    max_features):
    start_time = time.time()
    
    """前処理"""
    #ベクトル化するテキストファイルがあるフォルダのパスを取得
    file_paths = glob.glob(folder_name)
    print('全部のファイル数 = ', len(file_paths))
    
    #ファイルから読み込んだAPI関数群をリスト化(分かち書き前提)]
    document_tolist = [] #<--空のリストを用意。
    for file_path in file_paths[:]:
        with open(file_path, 'r') as f:
            file_tolist = f.read().split(' ')
            document_tolist.append(str(file_tolist)) #<--str型にしないと後でエラーになる。
    print('document_tolistの数 =', len(document_tolist)) 
    
    
    """モデル作成"""
    vectorizer = TfidfVectorizer(ngram_range=(n_gram, n_gram), max_features=max_features)
    values = vectorizer.fit_transform(document_tolist).toarray()
    words = vectorizer.get_feature_names()
    print('values.shape = ', values.shape)
    print('len(words)  = ', len(words))
    print('Success!!')
    
    
    """データフレーム化"""
    #ファイルの名前をリストに追加
    file_names = [] #<--データフレームにするときのindex_col
    for file_name in file_paths[:]:
        file_name = str(file_name)
        name = file_name.replace('./report8_dataset\\', '')
        file_names.append(name)
    print(len(file_names))

    df = pd.DataFrame(values, columns=words, index=file_names)
    csv_name = './CSV/Binarization/grad/tfidf/tfidf_ngram' + str(n_gram) + 'size' + str(max_features) + 'Trojan.csv'
    df.to_csv(csv_name)
    print('データフレームのshape', df.shape)
    print('すべて完了！！')
    
    #作成したcsvに正解ラベルを付ける。
    df = pd.read_csv(csv_name, index_col=0)

    """
    Nonevirus : 0
    Backdoor  : 1
    Ransom    : 2
    Trojan    : 3
    """

    for index_name in df.index[:]:
        if 'Trojan' in index_name:
            df.loc[index_name, 'LABEL'] = int(0)
        # elif 'Backdoor' in index_name:
        #     df.loc[index_name, 'LABEL'] = int(1)
        # elif 'Ransom' in index_name:
        #     df.loc[index_name, 'LABEL'] = int(2)
        else:
            df.loc[index_name, 'LABEL'] = int(1)
    df.to_csv(csv_name)
    
    end_time = time.time()
    return end_time - start_time
    
    
    
    
if __name__ == '__main__':
    
    
    folder_name = './report8_dataset/*txt' #<--ベクトル化するファルダの名前（正規表現）
    n_gram = 4 #<--n_gramの値
    max_features = 100 #<--最大特徴量
    result = main(folder_name, n_gram, max_features)
    print('---ベクトル化時間--')
    print(result, 's')
    
