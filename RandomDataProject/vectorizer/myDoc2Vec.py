"""
RandomDataProject用のDoc2VEcベクトル化プログラム
ベースディレクトリ=graduation
"""
import time
import glob
import json
import pandas as pd
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec

summary_key_lists = ["command_line", 
                    "connects_host",
                    "connects_ip",
                    "directory_created",
                    "directory_enumerated",
                    "directory_removed",
                    "dll_loaded",
                    "downloads_file",
                    "fetches_url",
                    "file_copied",
                    "file_created",
                    "file_deleted",
                    "file_exists",
                    "file_failed",
                    "file_moved",
                    "file_opened",
                    "file_read",
                    "file_recreated",
                    "file_written",
                    "guid",
                    "mutex",
                    "regkey_deleted",
                    "regkey_opened",
                    "regkey_read",
                    "regkey_written",
                    "resolves_host",
                    "tls_master",
                    "wmi_query"]

# Doc2Vecモデル用のパラメータ #
parameters = {
        'vector_size' : 100,
        'window_size' : 8,
        'iter'        : 500,
        'alpha'       : 0.1,
        'min_alpha'   : 0.001,
        'dm'          : 0,
        'seed'        : 4,
    }
class EpochLogger(CallbackAny2Vec):
    """訓練時のログを表示するためのクラス"""
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print(f"Epoch {self.epoch + 1} start")

    def on_epoch_end(self, model):
        print(f"Epoch {self.epoch + 1} end")
        self.epoch += 1


def get_summaryinfo(f_json):
    """
    Summary情報を取得する関数
    """
    summary_list = []
    for key in summary_key_lists:
        summary_list.append(f_json["summary"][key])
    summary_list.append(f_json["parent_count"])
    summary_list.append(f_json["children_count"])
    return summary_list




def Doc2VecMain(csv_num):
    """
    メイン関数 
    """

    # ランダムに選択したマルウェアのパスを取得する #
    malwarepaths = []
    with open(f"RandomDataProject/LOG/result{csv_num}_malwarepath.txt", mode="r", encoding="utf-8") as pathfile:
        for path in pathfile:
            malwarepaths.append(path.replace("\n", "")) #改行を取り除く

    All_APIlist = []
    All_indexlist = []
    All_summarylist = []
    tagged_corpus = [] #Doc2Vec用



    # クリーンウエアの情報を抽出する # 
    print("クリーンウェアのデータを抽出する")
    cleanware_paths = glob.glob("custom_datasets/RandomDataset/clean/*")
    for path in tqdm(cleanware_paths, desc="クリーンウェア"):
        tmp_string = ""
        tmp_summary = []
        with open(path, mode="r", encoding="utf-8") as cleanfile:
            CleanDict = json.load(cleanfile)

            # APIをTFIDF用に連結してリストに追加#
            for _, value in enumerate(CleanDict["all_api"]):
                tmp_string += str(value) + " "
            All_APIlist.append(tmp_string)

            # Summary情報をリストに追加 #
            tmp_summary = get_summaryinfo(CleanDict)
            All_summarylist.append(tmp_summary)

            #インデックス名を取得し格納 #
            tmp_index = path.replace("custom_datasets/RandomDataset/clean\\", "")
            All_indexlist.append(tmp_index)

            # コーパスに追加 #
            tagged_corpus.append(TaggedDocument(tmp_string, tmp_index))



    # マルウェアの情報を取得する # 
    print("マルウェアの情報を抽出する")
    for path in tqdm(malwarepaths, desc="マルウェア"):
        tmp_string = ""
        tmp_summary = []
        with open(path, mode="r", encoding="utf-8") as malwarefile:
            MalwareDict = json.load(malwarefile)

            # APIをスペースを空けて連結し、リストに追加#
            for _, value in enumerate(MalwareDict["all_api"]):
                tmp_string += str(value) + " "
            All_APIlist.append(tmp_string)

            # Summary情報をリストに追加 #
            tmp_summary = get_summaryinfo(MalwareDict)
            All_summarylist.append(tmp_summary)

            #インデックス名を取得し格納 #
            tmp_index = path.replace("custom_datasets/RandomDataset/malware\\", "")
            All_indexlist.append(tmp_index)

            #コーパスに追加 #
            tagged_corpus.append(TaggedDocument(tmp_string, tmp_index))
    

    # DocVecモデルの宣言 #
    epoch_logger = EpochLogger()

    model = Doc2Vec(
        tagged_corpus,
        vector_size= parameters['vector_size'],
        window = parameters['window_size'],
        # iter = parameters['iter'],
        dm = parameters['dm'],
        alpha = parameters['alpha'],
        min_alpha = parameters['min_alpha'],
        seed = parameters['seed'],
        callbacks=[epoch_logger],
    )

    # 訓練開始 #
    print("訓練開始")
    model.build_vocab(tagged_corpus)  # コーパスを基に単語辞書を構築
    start_time = time.time() #開始
    model.train(tagged_corpus, total_examples=model.corpus_count, epochs=parameters['iter'])  # 訓練実行
    train_end_time = time.time() #終了

    # 処理時間記録 #
    with open(f"experiment/RandomDataProject/TimeLog/doc2vec/result{csv_num}_Time.txt", mode="w", encoding="utf-8") as timefile:
        print(f"モデルの構築時間 = {train_end_time - start_time}", file=timefile)

        # ベクトル化した値を格納 #
        vectorized_data = []
        for i in range(len(tagged_corpus)):
            vectorized_data.append(model.infer_vector(All_APIlist[i]))
        
        # 計測終了#
        end_time = time.time()
        print(f"合計処理時間 = {end_time - start_time}", file=timefile)

    # データフレーム化 # 
    vectorized_df = pd.DataFrame(vectorized_data, index=All_indexlist)

    # Summary情報を付与する #
    summary_parent_children_columns = summary_key_lists + ["parent", "children"]
    summary_df = pd.DataFrame(All_summarylist, index=All_indexlist, columns=summary_parent_children_columns)
    result_df = pd.concat((vectorized_df, summary_df), axis=1)

    # LABELを付ける #
    result_df["LABEL"] = result_df.index.to_series().apply(lambda x: 0 if 'report' in x else 1)

    # CSV保存 #
    result_df.to_csv(f"CSV/RandomDatasetCSV/doc2vec/result{csv_num}.csv")

    




if __name__ == "__main__":
    # for i in range(10):
    #     Doc2VecMain(i+1)
    Doc2VecMain(1)