"""
PyTorchを使ってTLSHのCSVファイルを読み込んで
学習してみる。
"""

import torch
import pandas as pd


"""メイン関数"""
def main(csv: pd.DataFrame, dimention: int) -> None:
    print('This is main!') 


if __name__ == "__main__":
    print('#####This program learns TLSH using PyTorch!#####')

    target_csv = pd.read_csv("CSV/anything/tlsh_csv_doc2vec_2label.csv")
    print("ターゲットCSVのシェイプ = ", target_csv.shape)

    #次元数の設定
    dimention = target_csv.shape[1] -2
    print('入力データのデータ数 = ', target_csv.shape[0])
    print("入力データの次元数 = ", dimention)
    main(target_csv, dimention)
