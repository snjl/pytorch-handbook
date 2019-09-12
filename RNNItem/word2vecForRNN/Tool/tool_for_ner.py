import requests
import json


# 限定在排名最高的2个
def get_content_ner(sentence):
    a = requests.post('http://zhner.snjl.xyz/s', data={'s': sentence})

    j = json.loads(a.text)
    for key, value in j.items():
        j[key] = value[:2]
    return j

def get_content_ner_to_list(sentence):
    a = requests.post('http://zhner.snjl.xyz/s', data={'s': sentence})
    ner_list = list()
    j = json.loads(a.text)
    for key, value in j.items():
        # if key == "PER":
        ner_list.extend(value[:2])
    return ner_list


def is_part_of_ner(main_ner_list,ner_list,t):
    num = len(set(main_ner_list) & set(ner_list))
    if num >= t:
        return num
    else:
        return False

# while True:
#     s1 = input('sentence:')
#     s2 = input('sentence2:')
#     a = get_content_ner_to_list(s1)
#     b = get_content_ner_to_list(s2)
#     print(is_part_of_ner(a,b,2))
