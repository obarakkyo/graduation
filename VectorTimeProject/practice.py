import json

def check():
    target_path = "../特徴量エンジニアリング用/customdataset/FFRI2017_report/ML.Attribute.HighConfidence-28e05b9d8a1af2ac620f84f526963f642d11cb78137a9877402678f775c1e749.json"
    with open(target_path, mode="r", encoding="utf-8") as f:
        file_dict = json.load(f)
        tmp_API_List = file_dict["all_api"]
        print(f"len={len(tmp_API_List)}")


if __name__ == "__main__":
    check()