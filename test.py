import glob
import tlsh

def calcurate_tlsh(path: str) -> str:
    with open(path, 'rb') as file:
        result = tlsh.hash(file.read())
    return result

if __name__ == "__main__":
    nonmalware_file_paths = glob.glob('custom_datasets/dataset_1/report*')
    backdoor_file_paths = glob.glob('custom_datasets/dataset_1/Backdoor*')

    print('クリーンウェアの数：', len(nonmalware_file_paths))
    print('Backdoorの数:', len(backdoor_file_paths))

    # clearn_target_file_path = nonmalware_file_paths[0]
    # backdoor_target_file_path = backdoor_file_paths[0]

    # print(clearn_target_file_path, ' = ', calcurate_tlsh(clearn_target_file_path))
    print('##########バックドアのTLSH ##########')
    for file_path in backdoor_file_paths[:100]:
        print(calcurate_tlsh(file_path))

    print()

    print('##########クリーンウェアのTLSH ##########')
    for file_path in nonmalware_file_paths[:100]:
        print(calcurate_tlsh(file_path))

