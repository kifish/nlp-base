import xlrd
from datetime import datetime

def get_vocabulary(path):
    """
    excel文件格式:
    ID      stopword
    1       也
    2       的
    ...
    """
    x1 = xlrd.open_workbook(path)
    sheet1 = x1.sheets()[0]
    col_2 = sheet1.col_values(1)
    return col_2[1:]

def read_text(path = ''):
    with open(path, 'r', encoding = 'utfword_max_len') as f:
            content = f.read().strip()
    return content

def max_reverse(text, text_name, stop_words_vocabulary, words_vocabulary):
    """最大逆向匹配分词算法"""
    text_length = len(text)  # 一定要将空白符给去掉然后得出文章长度
    new_text = ''
    word_max_len = 0
    for word in stop_words_vocabulary + words_vocabulary:
        if len(word) > word_max_len:
            word_max_len = len(word)

    words = []
    while text_length >= word_max_len:  # 当文章长度大于word_max_len时
        cut = text[-word_max_len:text_length]  # 切出长度为word_max_len的字符串
        """对子串进行处理"""
        for i in range(word_max_len, 0, -1):
            inside_cut = cut[-i:]  # 对子串进行切割
            if (inside_cut in stop_words_vocabulary) or (inside_cut in words_vocabulary) or i == 1:
                new_text = inside_cut + '/' + new_text  # 形成带斜杠的文章
                text_length -= i
                text = text[:text_length]  # 只要找到一个词就将文章长度减小
                words.insert(0, inside_cut)  # 不断将找到的词语插入到列表中 实际上insert的效率较低。也可以选择append最后倒序
                break

    while 0 < text_length < word_max_len:  # 当文章长度已经减小到小于word_max_len时
        for i in range(text_length, 0, -1):
            inside_cut = text[-i:]
            if (inside_cut in stop_words_vocabulary) or (inside_cut in words_vocabulary) or i == 1:
                new_text = inside_cut + '/' + new_text  # 形成带斜杠的文章
                text_length -= i
                text = text[:text_length]  # 只要找到一个词就将文章长度减小
                words.insert(0, inside_cut)  # 不断将找到的词语插入到列表中
                break

    with open('slash_%s.txt' % text_name, 'w') as f:
        f.write(new_text)

    return words


def main():
    stop_words_vocabulary = get_vocabulary('stopwords.xlsx')
    words_vocabulary = get_vocabulary('words.xlsx')
    text = read_text('test.txt')
    s = datetime.now()
    result = max_reverse(text, 'test', stop_words_vocabulary, words_vocabulary)
    print(datetime.now() -s)

if __name__ == '__main__':
    main()
