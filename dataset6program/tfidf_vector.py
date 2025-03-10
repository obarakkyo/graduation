"""
API先頭100個をTF-IDFを用いてベクトル化したCSVを出力する.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import time


def training_thidf(all_api_string, index_list, n_gram=1):
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(n_gram, n_gram))
    vectorizer.fit(all_api_string)

    tmp_np = vectorizer.transform(all_api_string).toarray()
    columns = vectorizer.get_feature_names_out()
    vectorized_df = pd.DataFrame(tmp_np, columns=columns, index=index_list)
    return vectorized_df

def main():
    # データ取得 #
    get_csvpath = "../CSV/dataset6CSV/origin/2label.csv"
    df = pd.read_csv(get_csvpath, index_col=0)
    print(f"csvpath={get_csvpath}\ndf.shape={df.shape}")

    # APIを取り出す #
    api_list = df.iloc[:, 0:100].reset_index(drop=True).to_numpy().tolist()

    # 空白文字列に変換 #
    all_api_string = []
    for current_list in api_list:
        tmp_string = " ".join(current_list)
        all_api_string.append(tmp_string)

    #Tf-idfでベクトル化
    n_gram = 3
    start_time = time.time()
    vectorized_df = training_thidf(all_api_string, index_list=df.index, n_gram=n_gram) #(2511, 100)
    end_time = time.time()


    # 実行時間をファイルに保存 #
    saving_file_path = f"../experiment/vectorizer_time/tfidf/{n_gram}gram/time.txt"
    with open(saving_file_path, mode="w", encoding="utf-8") as f:
        print(f"ベクトル化実行時間 = {end_time - start_time}", file=f)


    #Summary列の連結
    tmp_df = df.iloc[:, 100:]
    tmp_df = pd.concat([vectorized_df, tmp_df], axis=1)
    print(tmp_df.shape)

    #保存
    # save_path = f"../CSV/dataset6CSV/tfidf/max100_{n_gram}gram_2label.csv"
    # tmp_df.to_csv(save_path)


    
    
    
    
    

if __name__ == "__main__":
    main()