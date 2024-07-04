""" プログラムの説明

すでに作成されたTLSHのＣＳＶファイルを
Doc2Vecによるベクトル化をして
新しいCSVを作成する。

"""

"""必要なライブラリのインポート"""
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import glob
import time


def change_to_doc2vec(csv_path: str, parameters: dict, csv_save_path: str) -> None:
    df = pd.read_csv(target_csv, index_col=0)

    hash_lists = df.iloc[:, 0:-2].values.tolist()

    tags = df.index.to_list()

    tagged_corpus = []

    ## １文字以外の時に使う ##
    # for i in range(len(hash_lists)):
    #     tagged_corpus.append(TaggedDocument(hash_lists[i], tags[i]))

    ## １文字の時 ##
    for i in range(len(hash_lists)):
        #文字列に変換
        hash_list_str = [str(word) for word in hash_lists[i]]
        tagged_corpus.append(TaggedDocument(hash_list_str, tags[i]))
    
    """モデルの作成"""
    print('-----Creating Model-----')
    print('vector_size = ', parameters['vector_size'])
    print('window_size = ', parameters['window_size'])
    print('iter = ', parameters['iter'])
    print('alpha', parameters['alpha'])
    print('min_alpha', parameters['min_alpha'])
    print('dm = ', parameters['dm'])
    print('seed', parameters['seed'])

    model = Doc2Vec(
        tagged_corpus,
        vector_size= parameters['vector_size'],
        window = parameters['window_size'],
        # iter = parameters['iter'],
        dm = parameters['dm'],
        alpha = parameters['alpha'],
        min_alpha = parameters['min_alpha'],
        seed = parameters['seed'],
    )
    # model_name = 'Mydoc2vecModel/tlsh/s' + str(parameters['vector_size']) + 'w' + str(parameters['window_size']) + 'alpha' + str(parameters['alpha']) + '-' + str(parameters['min_alpha']) + 'dm' + str(parameters['dm']) + 'seed' + str(parameters['seed']) + '.model'
    # model.save(model_name)


    """CSVを作成する"""
    vector_lists = []
    for i in range(len(tagged_corpus)):
        hash_list_str = [str(word) for word in hash_lists[i]]
        vector_lists.append(model.infer_vector(hash_list_str))
    
    vector_df = pd.DataFrame(vector_lists, index=tags)

    ### ラベル付け4値 ###
    """
    Backdoor.Graybird 1
    Packed.Generic    2
    Ransom.Cerber     3
    report            0
    """
    # for index_name in vector_df.index[:]:
    #     if index_name.startswith("report"):
    #         vector_df.loc[index_name, 'LABEL'] = 0
    #     elif index_name.startswith("Backdoor"):
    #         vector_df.loc[index_name, 'LABEL'] = 1
    #     elif index_name.startswith("Packed"):
    #         vector_df.loc[index_name, 'LABEL'] = 2
    #     elif index_name.startswith("Ransom"):
    #         vector_df.loc[index_name, 'LABEL'] = 3
    #     else:
    #         print("ラベル付けできないファイルがあります。")
    #         exit()
    
    ###ラベル付け　２値
    """
    Backdoor.Graybird 1
    Packed.Generic    1
    Ransom.Cerber     1
    report            0
    """
    for index_name in vector_df.index[:]:
        if index_name.startswith("report"):
            vector_df.loc[index_name, 'LABEL'] = 0
        else:
            vector_df.loc[index_name, 'LABEL'] = 1
    
    ###ラベル付け　６値
    """
    report               0
    Backdoor.Graybird    1
    Packed.Generic       2
    Ransom.Cerber        3
    Infostealer.Limitail 4
    Trojan.Gen           5
    """
    # for index_name in vector_df.index[:]:
    #     if index_name.startswith("report"):
    #         vector_df.loc[index_name, 'LABEL'] = 0
    #     elif index_name.startswith("Backdoor"):
    #         vector_df.loc[index_name, 'LABEL'] = 1
    #     elif index_name.startswith("Packed"):
    #         vector_df.loc[index_name, 'LABEL'] = 2
    #     elif index_name.startswith("Ransom"):
    #         vector_df.loc[index_name, 'LABEL'] = 3
    #     elif index_name.startswith("Info"):
    #         vector_df.loc[index_name, 'LABEL'] = 4
    #     elif index_name.startswith("Trojan"):
    #         vector_df.loc[index_name, 'LABEL'] = 5
    #     else:
    #         print("ラベル付けできないファイルがあります。")
    #         print('index_name = ', index_name)
    #         exit()


    
    vector_df.to_csv(csv_save_path)



if __name__ ==  "__main__":
    print("This program changes tlsh to doc2vec value!")

    """dataset1"""
    # target_csv = "CSV/anything/tlsh_csv_origin_1_spilit_4label.csv"
    # target_csv = "CSV/anything/tlsh_csv_origin_2_spilit_4label.csv"
    # target_csv = "CSV/anything/tlsh_csv_origin_3spilit_4label.csv"
    # target_csv = "CSV/anything/tlsh_csv_origin_4spilit_4label.csv"

    """dataset2"""
    # target_csv = "../CSV/dataset2CSV/origin/tlsh_csv_origin_1_spilit_6label.csv"
    # target_csv = "../CSV/dataset2CSV/origin/tlsh_csv_origin_2_spilit_6label.csv"
    # target_csv = "../CSV/dataset2CSV/origin/tlsh_csv_origin_3_spilit_6label.csv"
    target_csv = "../CSV/dataset2CSV/origin/tlsh_csv_origin_4_spilit_6label.csv"



    #Doc2Vecのパラメータを指定
    parameters = {
        # 'vector_size' : 100,
        # 'vector_size' : 18,
        'vector_size' : 18,
        'window_size' : 8,
        'iter'        : 500,
        'alpha'       : 0.1,
        'min_alpha'   : 0.001,
        'dm'          : 0,
        'seed'        : 4,
    }

    #CSVとして保存する場所
    csv_save_path = '../CSV/dataset2CSV/doc2vec/tlsh_doc2vec_4split_18dim_2label.csv'

    change_to_doc2vec(target_csv, parameters, csv_save_path)