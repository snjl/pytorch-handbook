import Tool.tool_for_cut_text as tool_for_cut_text
import numpy as np


# 句子得到jieba分词后的list，默认使用停用词表，暂时由自己构建
def get_cut_sentence(text, stop_words=True):
    if stop_words is True:
        return tool_for_cut_text.cut_words_by_line_to_list_with_stop_words(text)
        # return tool_for_cut_text.cut_words_by_line_to_list_without_len_less_2(text)
    else:
        return tool_for_cut_text.cut_words_by_line_to_list(text)


# 得到句子的向量
def get_sentence_words_vector(get_sentence, w2v_model):
    num = len(get_sentence)
    # print("the sentence is %s" % get_sentence)
    average_vector = np.zeros((w2v_model.vector_size,))
    for word in get_sentence:
        if word != ' ':
            try:
                word_vector = w2v_model[word]
                average_vector += word_vector
            except KeyError:
                # print("%s don't exist in vocabulary" % word)
                num -= 1
    if num == 0:
        return np.zeros((w2v_model.vector_size,))
    else:
        average_vector = average_vector / num
        # print("num : %s" % num)
        return average_vector


# 得到句子的向量
def get_sentence_words_vector_to_rnn(get_sentence, w2v_model):
    num = len(get_sentence)
    # print("the sentence is %s" % get_sentence)
    rnn_vector = list()
    for word in get_sentence:
        if word != ' ':
            try:
                word_vector = w2v_model[word]
                rnn_vector.append(word_vector)
            except KeyError:
                # print("%s don't exist in vocabulary" % word)
                num -= 1
    if num == 0:
        return np.zeros((w2v_model.vector_size,))
    else:
        rnn_vector = np.array(rnn_vector)
        return rnn_vector

# 得到keywords关键词集合的向量，keywords数量大于等于3时，采用权重计算方法
def get_keywords_vector(get_sentence, w2v_model):
    num = len(get_sentence)
    real_sentence = list()
    for word in get_sentence:
        try:
            a = w2v_model[word]
            if a is not None:
                real_sentence.append(word)
        except KeyError:
            num -= 1

    average_vector = np.zeros((w2v_model.vector_size,))
    if num < 3:
        for word in real_sentence:
            if word != ' ':
                try:
                    word_vector = w2v_model[word]
                    average_vector += word_vector
                except KeyError:
                    # print("%s don't exist in vocabulary" % word)
                    num -= 1
        if num == 0:
            return np.zeros((w2v_model.vector_size,))
        else:
            average_vector = average_vector / num
            # print("num : %s" % num)
            return average_vector
    else:
        average_percent = 0.7 /(len(real_sentence) - 3)
        for i, word in enumerate(real_sentence):
            if word != ' ':
                if i < 3:
                    try:
                        word_vector = w2v_model[word]
                        average_vector += word_vector * 0.1
                    except KeyError:
                        # print("%s don't exist in vocabulary" % word)
                        num -= 1
                        i -= 1
                else:
                    word_vector = w2v_model[word]
                    average_vector += word_vector * average_percent
        if num == 0:
            return np.zeros((w2v_model.vector_size,))
        else:
            return average_vector


# 得到两个向量的余弦相似度
def get_cosine_similarity(vector_1, vector_2):
    return np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * (np.linalg.norm(vector_2)))


# 测试两个句子的相似度，通过get_sentence_words_vector得到两个句子的向量，调用get_consine_similarity得到相似度
def get_similarity_by_cosine_similarity(model):
    while True:
        sentence = input("sentence:")
        sentence = get_cut_sentence(sentence)
        vector1 = get_sentence_words_vector(sentence, model)
        sentence = input("sentence:")
        sentence = get_cut_sentence(sentence)
        vector2 = get_sentence_words_vector(sentence, model)
        sentence_similarity = get_cosine_similarity(vector1, vector2)
        print("the similarity is :%s" % sentence_similarity)
        if sentence == " ":
            break


# if __name__ == '__main__':
    # path = tool_for_word2vec_test.choose_model()
    # model = w2v.Word2Vec.load("news_model\\model1\\model1.model")
    # while True:
    #     sentence = input("sentence:")
    #     sentence = get_cut_sentence(sentence)
    #     vector_1 = get_sentence_words_vector(sentence, model)
    #     sentence = input("sentence:")
    #     sentence = get_cut_sentence(sentence)
    #     vector_2 = get_sentence_words_vector(sentence, model)
    #     print("the similarity is :%s" % get_similarity_by_cosine_similarity(vector_1, vector_2))
    # a = np.array(model["你好"])
    # print(a)
    # b = np.array(model["再见"])
    # c = np.array(model["相似"])
    # print("你好 再见 ： %s  " % model.similarity("你好", "再见"))
    # print("你好 相似 ： %s" % model.similarity("你好", "相似"))
    # print(np.dot(a, c)/(np.linalg.norm(a)*(np.linalg.norm(c))))

