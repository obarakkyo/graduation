"""
dataset6を対象にDoc2vecによるベクトル化をするプログラム。
"""
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time



"""Doc2Vecによるベクトル化関数"""
def df_todoc2vec(normal_df, parameters):
    values_list = normal_df.values.tolist() #df -> list
    tags = normal_df.index.to_list()

    tagged_corpus = []

    """コーパスの作成"""
    for i in range(normal_df.shape[0]):
        tagged_corpus.append(TaggedDocument(values_list[i], tags[i]))
    
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

    # モデルの訓練
    model.build_vocab(tagged_corpus)  # コーパスを基に単語辞書を構築
    model.train(tagged_corpus, total_examples=model.corpus_count, epochs=parameters['iter'])  # 訓練実行


    

    """データフレームに変換"""
    vectorized_list = []
    for i in range(len(tagged_corpus)):
        vectorized_list.append(model.infer_vector(values_list[i]))
    
    vectorized_df = pd.DataFrame(vectorized_list, index=tags)
    return vectorized_df





def main():
    targetcsv_path = "../CSV/dataset6CSV/origin/2label.csv"
    saveing_path = "../CSV/dataset6CSV/doc2vec/2label.csv"

    change_colmax = 100

    parameters = {
        'vector_size' : 100,
        'window_size' : 8,
        'iter'        : 500,
        'alpha'       : 0.1,
        'min_alpha'   : 0.001,
        'dm'          : 0,
        'seed'        : 4,
    }

    """CSVの読み込み"""
    df = pd.read_csv(targetcsv_path, index_col=0)
    print("df.shape = {}".format(df.shape))


    """ベクトル化対象列抜き出し"""
    normal_df = df.iloc[:, 0:change_colmax]
    print("normal.shape = {}".format(normal_df.shape))


    """Doc2Vecによるベクトル化処理"""
    start_time = time.time()
    changed_df = df_todoc2vec(normal_df, parameters)
    end_time = time.time()

    """ファイルに実行時間を保存"""
    savign_file_path = f"../experiment/vectorizer_time/doc2vec/time.txt"
    with open(savign_file_path, mode="w", encoding="utf-8") as f:
        print(f"ベクトル化実行時間 = {end_time - start_time}", file=f)
    
    """データフレームに上書きする"""
    df.iloc[:, 0:change_colmax] = changed_df
    print(df)

    """CSVとして保存する"""
    df.to_csv(saveing_path)

if __name__ == "__main__":
    main()