import os
import jieba
from tqdm import tqdm

"""
data preprocessing
"""


def get_token(path='train'):
    """
    在训练集中取出所有词汇，制作字典
    UNK:  Unknown words
    PAD:  Padding

    :param path:
    :return:
    """
    token = []
    for sample_type in ['neg', 'pos']:
        for filename in tqdm(os.listdir(os.path.join(path, sample_type)), desc='Get word in train datasets'):
            fr = open(os.path.join(path, sample_type, filename), encoding='utf-8')
            for sentence in fr.readlines():
                sentence = sentence.strip()
                sentence = jieba.lcut(sentence)
                for word in sentence:
                    word = word.lower()
                    word = word.strip()

                    if (len(word) == 0) or (len(word) > 45):
                        continue

                    if word not in token:
                        token.append(word)
            fr.close()

    with open('data/user_dict.txt', 'w', encoding='utf-8') as fw:
        print('saving...')
        fw.write('UNK\n')
        fw.write('PAD\n')
        for word in tqdm(token):
            fw.write(word + '\n')


def word_index(file_dir, user_dict_path):
    """
    将训练集和测试集的词汇转为字典中的索引
    :param file_dir: e.g. aclImdb/train
    :param user_dict_path:
    :return:
    """
    with open(user_dict_path, 'r', encoding='utf-8') as f_dict:
        user_dict = [str(i).strip() for i in f_dict.readlines()]

    with open(os.path.join('data', (os.path.split(file_dir)[-1] + '.txt')), 'w', encoding='utf-8') as fw:

        for sample_type in ['neg', 'pos']:
            for filename in tqdm(os.listdir(os.path.join(file_dir, sample_type)), desc=sample_type):
                f_s = open(os.path.join(file_dir, sample_type, filename), 'r', encoding='utf-8')

                if sample_type == 'neg':
                    fw.write('0 ')
                else:
                    fw.write('1 ')

                for sentence in f_s.readlines():
                    sentence = sentence.strip()
                    sentence = jieba.lcut(sentence)
                    for word in sentence:
                        word = word.lower()
                        word = word.strip()

                        if len(word) == 0:
                            continue

                        if word in user_dict:
                            fw.write(str(user_dict.index(word)) + ' ')
                        else:
                            fw.write('0 ')  # Unknown words' index is 0

                fw.write('\n')

                f_s.close()


if __name__ == '__main__':
    word_index(r'D:\Datasets\IMDB\aclImdb_v1.tar\aclImdb_v1\aclImdb\train', 'data/user_dict.txt')
    word_index(r'D:\Datasets\IMDB\aclImdb_v1.tar\aclImdb_v1\aclImdb\test', 'data/user_dict.txt')
