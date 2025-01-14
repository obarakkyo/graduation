import json

def check():
    tmp_list = []
    api_list = ["abc", "def", "ghi"]

    tmp_list.append(" ".join(api_list))
    tmp_list.append(" ".join(api_list))

    print(tmp_list)
    
    


if __name__ == "__main__":
    check()