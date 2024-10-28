"""
API先頭100個をTF-IDFを用いてベクトル化したCSVを出力する.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def training_thidf(all_api_string):
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 1))
    vectorizer.fit(all_api_string)

    tmp_np = vectorizer.transform(all_api_string).toarray()
    columns = vectorizer.get_feature_names_out()
    df = pd.DataFrame(tmp_np, columns=[columns])
    df.to_csv("tmp.csv")
    return None

def main():
    # データ取得 #
    get_csvpath = "../CSV/dataset6CSV/origin/2label.csv"
    df = pd.read_csv(get_csvpath, index_col=0)
    print(f"csvpath={get_csvpath}\ndf.shape={df.shape}")

    # APIを取り出す #
    api_list = df.iloc[:, 0:100].reset_index(drop=True).to_numpy().tolist()
    # print(api_list[0])

    # 空白文字列に変換 #
    all_api_string = []
    for current_list in api_list:
        tmp_string = " ".join(current_list)
        all_api_string.append(tmp_string)

    training_thidf(all_api_string)
    
    
    
    
    

if __name__ == "__main__":
    main()