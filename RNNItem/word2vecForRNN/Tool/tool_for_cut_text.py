import os
import Tool.tool_for_replace as tool_for_replace
import jieba


# 将文件夹内的文件进行去除符号，停用词，分词处理（仅限单层文件夹）
def cut_text(from_dir, to_dir, num=True, letter=True, stop_words=True):
    for parent, dirnames, filenames in os.walk(from_dir):
        # 将text文件夹中的txt文档用jieba进行分词处理
        print("enter folder:" + from_dir)
        for filename in filenames:
            file_position = from_dir + "\\" + filename
            print("start processing:" + filename)
            # file_position 是文件路径
            fr = open(file_position, "r", encoding="gb18030", errors="ignore")
            replace_list = tool_for_replace.replace_txt(fr.read(), num, letter)
            if stop_words is True:
                # 去除停用词
                stop_words_list = load_stop_words_list_with_space("stop_words.txt")
                for word in fr.readlines():
                    word = word.replace("\n", "")
                    stop_words_list.append(word)
                fr.close()
                replace_list = tool_for_replace.replace_stop_words(replace_list, *stop_words_list)

            seg_list = jieba.cut(replace_list)
            cut_file_positon = to_dir + "\\cut_" + filename
            fw = open(cut_file_positon, "w", encoding="utf8")
            # 存为utf8格式，方便后续相似度计算
            fw.write(" ".join(seg_list))
            print("file is stord:" + "cut_" + filename + "\n")
            fr.close()
            fw.close()


# 加载停用词表:停用词前面有空格，则在停用词后加入空格，表示是单独的字，如果没有空格，照常处理
def load_stop_words_list_with_space(stop_words_list_path="stop_words.txt"):
    fr = open(stop_words_list_path, 'r', encoding='utf8')
    stop_words_list = []
    for word in fr.readlines():
        word = word.replace("\n", "")
        if word != "" and word[0] == " ":
            word = word + " "
        stop_words_list.append(word)
    fr.close()
    return stop_words_list


# 加载停用词表，默认路径为项目目录的stop_words.txt
def load_stop_words_list(stop_words_list_path="stop_words.txt"):
    fr = open(stop_words_list_path, 'r', encoding='utf8')
    stop_words_list = []
    for word in fr.readlines():
        word = word.replace("\n", "")
        word = word.replace(" ", "")
        stop_words_list.append(word)
    fr.close()
    return stop_words_list


# 先对line进行分词，再去除停用词，采用带空格的停用词表，返回带空格的分词的句子
def cut_words_by_line_with_stop_words(string):
    stop_words = load_stop_words_list_with_space()
    string = jieba.cut(string)
    string = " ".join(string)
    string = tool_for_replace.replace_stop_words(string, *stop_words)
    string = string.replace("\n", "")
    string = " ".join(string.split())
    return string


# 仅对line去除一些字符和英文数字，进行分词，返回带空格的分词句子
def cut_words_by_line(string):
    string = tool_for_replace.replace_txt(string)
    string = jieba.cut(string)
    string = " ".join(string)
    string = string.replace("\n", "")
    string = " ".join(string.split())
    return string


# 对line去除一些字符和英文数字，进行分词后返回list类型的分词结果
def cut_words_by_line_to_list(line):
    line = tool_for_replace.replace_txt(line)
    line_l = jieba.lcut(line)
    return line_l


# 对line去除一些字符和英文数字，进行分词后加载停用词表(加载的普通停用词表，不带空格）后进行倒序删除，返回list类型分词结果
def cut_words_by_line_to_list_with_stop_words(line):
    line = tool_for_replace.replace_txt(line)
    stop_words = load_stop_words_list()
    line_l = jieba.lcut(line)
    for stop_word in stop_words:
        while stop_word in line_l:
            line_l.remove(stop_word)
    return line_l


# 对line去除一些字符和英文数字，得到list类型的分词结果，去除长度小于2的元素(简单粗暴)
def cut_words_by_line_to_list_without_len_less_2(line):
    # line = tool_for_replace.replace_txt(line)
    line_l = jieba.lcut(line)
    for s in reversed(line_l):
        if len(s) < 2:
            line_l.remove(s)
    return line_l


# 去除换行符(好像没什么用…)
def replace_n(string):
    return string.replace("\n", "")
