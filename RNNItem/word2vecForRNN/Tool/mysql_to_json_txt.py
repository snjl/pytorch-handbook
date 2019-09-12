import pymysql
import json
"""
将数据库数据存到tech.txt中，每一行为一个json，含有title、abstract、content、pubTime、keywords字段

"""

def item_to_json(d):
    # d = dict()
    # d['title'] = title
    # d['abstract'] = abstract
    # d['content'] = content
    # d['keywords'] = keywords
    return json.dumps(d, ensure_ascii=False)


connection = pymysql.connect(host='localhost', user='root', password='123456',
                             database='tech', charset='utf8')

cursor = connection.cursor()

cursor.execute("select * from xinhua")

infos = cursor.fetchall()
for info in infos:
    info_dict = dict()
    info_dict['title'] = info[0]
    info_dict['abstract'] = info[3]
    info_dict['content'] = info[8]
    info_dict['keywords'] = info[4]
    info_dict['pubTime'] = str(info[6])
    info_json = item_to_json(info_dict)
    with open('tech.txt', 'a+', encoding='utf8', errors='ignore') as f:
        f.write(info_json)
        f.write("\n")
# print(cursor.fetchall())
