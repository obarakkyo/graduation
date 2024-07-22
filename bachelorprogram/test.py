import glob
import tlsh
import pandas as pd

### ファイルパスを受け取って、TLSHのハッシュ値を返す ###
def calcurate_tlsh(path: str) -> str:
    with open(path, 'rb') as file:
        result = tlsh.hash(file.read())
    return result


### main関数　###
if __name__ == "__main__":
    # nonmalware_file_paths = glob.glob('custom_datasets/dataset_1/report*')
    backdoor_file_paths = glob.glob('custom_datasets/dataset_1/Backdoor*')


    # print('クリーンウェアの数：', len(nonmalware_file_paths))
    print('Backdoorの数:', len(backdoor_file_paths))

    print('##########バックドアのTLSH ##########')
    tlsh_list = []
    index_name = []
    for file_path in backdoor_file_paths[:100]:
        tlsh_list.append(calcurate_tlsh(file_path))
        modify_name =file_path.replace('custom_datasets/dataset_1\\', '')
        index_name.append(modify_name)

    print(len(tlsh_list))
    print(len(index_name))

    #データフレームに変換（まだラベルついてない）
    new_df = pd.DataFrame(tlsh_list, index=index_name)

    #LABEL付け
    for index_name in new_df.index[:]:
        if 'Backdoor' in index_name:
            new_df.loc[index_name, 'LABEL'] = 1
        else:
            new_df.loc[index_name, 'LABEL'] = 0

    print(new_df)
    

    # print('##########クリーンウェアのTLSH ##########')
    # for file_path in nonmalware_file_paths[:100]:
    #     tlsh_list.append(calcurate_tlsh(file_path))

