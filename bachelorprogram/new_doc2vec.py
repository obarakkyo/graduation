#必要なモジュールのインポート
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import glob
import pandas as pd
import time

def main(folder_path, filename_cut, parameters, csv_name):
    """タイム計測開始"""
    start_time = time.time()

    """対象のファイルのpathを一括取得"""
    file_paths = glob.glob(folder_path)
    print('全部のファイルの数 = ', len(file_paths))

    """データの前処理"""
    text_list = [] #<--全部に文書のリスト
    tags = [] #<--全文書のid格納用リスト

    tagged_cprpus = []

    for file_path in file_paths[:]:    #<-----------あとで変える！とりあえず10個
        with open(file_path, 'r') as f:
            #その文書の単語リストをtext_listに格納
            file_tolist = f.read().split(' ')
            text_list.append(file_tolist)

            #idを作る(fileの名前をidにする)
            file_name = file_path.replace(filename_cut, '')
            file_name = file_name.replace('.json.txt.txt', '')
            tags.append(file_name)

            tagged_cprpus.append(TaggedDocument(file_tolist, file_name))
    print('len(text_list)', len(text_list))
    print('len(tags) = ', len(tags))
    print()


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
        tagged_cprpus,
        vector_size= parameters['vector_size'],
        window = parameters['window_size'],
        # iter = parameters['iter'],
        dm = parameters['dm'],
        alpha = parameters['alpha'],
        min_alpha = parameters['min_alpha'],
        seed = parameters['seed'],
    )

    model_name = './models/doc2vec/s' + str(parameters['vector_size']) + 'w' + str(parameters['window_size']) + 'alpha' + str(parameters['alpha']) + '-' + str(parameters['min_alpha']) + 'dm' + str(parameters['dm']) + 'seed' + str(parameters['seed']) + '.model'
    model.save(model_name)

    """作成したモデルで文書のベクトル群をcsvに保存したい."""
    csv_list = []
    #モデル作成に用いたファイルの数だけ実行
    for i in range(len(tagged_cprpus)): 
        vector = model.infer_vector(text_list[i])
        csv_list.append(vector)
    
    #データフレームに変換する: 正常なプログラム 0 , マルウェア 1
    df = pd.DataFrame(csv_list, index=tags)
    
    for index_name in df.index[:]:
        if 'Trojan' in index_name:
            df.loc[index_name, 'LABEL'] = int(0)
        else:
            df.loc[index_name, 'LABEL'] = int(1)
    df.to_csv(csv_name)

    """タイム計測終了"""
    end_time = time.time()
    print()
    print('-------------Finished!!--------------')
    print('Time:', end_time - start_time)


    
    
    
if __name__ == '__main__':
    #対象データセットのpathを指定(正規表現)
    folder_path = './report8_dataset/*txt'
    filename_cut = './report8_dataset\\'

    #Doc2Vecのパラメータを指定
    parameters = {
        'vector_size' : 100,
        'window_size' : 8,
        'iter'        : 500,
        'alpha'       : 0.1,
        'min_alpha'   : 0.001,
        'dm'          : 0,
        'seed'        : 4,
    }

    #csvとして保存したいpathを指定
    csv_name = './CSV/Binarization/grad/doc2vec/Trojan_s' + str(parameters['vector_size']) + 'w' + str(parameters['window_size']) + 'alpha' + str(parameters['alpha']) + '-' + str(parameters['min_alpha']) + 'dm' + str(parameters['dm']) + 'seed' + str(parameters['seed']) + '.csv'
    main(folder_path, filename_cut, parameters, csv_name)