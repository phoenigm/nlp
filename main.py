import nltk as nltk
import pandas
import pymorphy2
from collections import Counter
import math
import numpy

analyzer = pymorphy2.MorphAnalyzer()
reviews_result = open("reviews-output.csv", "w+", encoding="UTF-8")
punctuation = [',', '.', ':', '?', '«', '»', '-', '(', ')', '!', '\'', '—', ';', '”', '...', '—', '!', '\'']


def normalized_reviews():
    df = pandas.read_excel("reviews-input-rating.xlsx", nrows=30, skiprows=60)
    df.columns = ["name", "text", "rating"]
    df = df["text"]
    normalized_text = []
    for row in df:
        words = nltk.word_tokenize(row)
        normalized_words = []
        for token in words:
            _word = analyzer.parse(token)[0]
            if token in punctuation or _word in punctuation:
                continue
            normalized_words.append(_word.normal_form)
        normalized_text.append(normalized_words)
    return normalized_text


def compute_tf_idf(corpus):
    def compute_tf(_text):
        tf_text = Counter(_text)
        for i in tf_text:
            tf_text[i] = tf_text[i] / float(len(_text))
        return tf_text

    def compute_idf(_word, _corpus):
        return math.log10(len(_corpus) / sum([1.0 for i in _corpus if _word in i]))

    documents_list = []
    for text in corpus:
        tf_idf_dictionary = {}
        computed_tf = compute_tf(text)
        for _word in computed_tf:
            tf_idf_dictionary[_word] = computed_tf[_word] * compute_idf(_word, corpus)
        documents_list.append(tf_idf_dictionary)
    return documents_list


tf_idf = compute_tf_idf(normalized_reviews())
print(tf_idf)
sorted_tf_idf_file = open("sorted_tfidf_file.txt", "w", encoding="UTF-8")

for dictionary in tf_idf:
    sorted_dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
    print(sorted_dictionary, sorted_tf_idf_file)


all_unique_words = []
for dictionary in tf_idf:
    for key in dictionary.keys():
        all_unique_words.append(key)
unique_words = numpy.unique(all_unique_words)
print(len(unique_words))


unique_dictionary = {}
for word in unique_words:
    unique_dictionary[word] = 0
    for dictionary in tf_idf:
        unique_dictionary[word] += dictionary.get(word, 0)
sorted_dictionary = {k: v for k, v in sorted(unique_dictionary.items(), key=lambda item: item[1], reverse=True)}
for word, rank in sorted_dictionary.items():
    print(word + ": " + str(rank))

