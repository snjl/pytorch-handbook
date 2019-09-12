import re


# 仅去除稀奇古怪的字符
def replace_txt_all(txt):
    txt = re.sub("[A-Za-z0-9①-⑨\[\`\~\!\@\？\【\】\—\☆\┃\#\$\^\&\*\(\)\=\|\{\}\'\:\：\；\;\'\,\[\]\."
                 "\<\>\/\，\。\、\“\，\：\》\《\…\。\`\·\” \?\~\！\!\$\%\-\_\（\）"
                 "\+\=\*\&\^\<\>\"\@\#\\\&\*\%]", "", txt)
    return txt


# 去除稀奇古怪的字符，默认去除数字和字母
def replace_txt(txt, num=True, letter=True):
    if num:
        num_string = "0-9"
    else:
        num_string = ""
    if letter:
        letter_string = "A-Za-z"
    else:
        letter_string = ""
    txt = re.sub("[" + letter_string + num_string + "①-⑨\[\`\~\!\@\？\【\】\—\☆\┃\#\$\^\&\*\(\)\=\|\{\}\'\:\：\；\;\'\,\[\]\."
                 "\<\>\/\，\。\、\“\，\：\》\《\…\。\`\·\” \?\~\！\!\$\%\-\_\（\）"
                 "\+\=\*\&\^\<\>\"\@\#\\\&\*\%]", "", txt)
    return txt


# 去除停用词，如果遇到第一个字符为空格，则表示为单独的字或者词，去掉后加一个空格
def replace_stop_words(txt, *stop_words):
    for stop_word in stop_words:
        if stop_word != "" and stop_word[0] == " ":
            txt = txt.replace(stop_word, " ")
        else:
            txt = txt.replace(stop_word, "")
    return txt
