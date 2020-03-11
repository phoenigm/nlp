import nltk as nltk
import numpy
import pandas
import pymorphy2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


def get_normal_form_of_single_text(text):
    analyzer = pymorphy2.MorphAnalyzer()
    punctuation = [',', '.', ':', '?', '«', '»', '-', '(', ')', '!', '\'', '—', ';', '”', '...', '—', '!', '\'']
    normalized_words = []
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        if token in punctuation: continue
        normalized_words.append(analyzer.parse(token)[0].normal_form)
    return normalized_words


def get_bag_of_words_vectors(all_texts, normalized_texts, vocabulary):
    vectors = []
    j = 0
    for text in normalized_texts:
        bag_vector = numpy.zeros(len(vocabulary))
        for w in text:
            for i, word in enumerate(vocabulary):
                if word == w:
                    bag_vector[i] += 1
        vectors.append(bag_vector)
        j += 1
        print("Progress: " + str(j) + "/" + str(len(all_texts)))
    return vectors


def get_vocab(normalized_texts):
    return numpy.unique(numpy.concatenate(normalized_texts))


def get_normalized_texts(all_texts):
    normalized_texts = []
    i = 0
    for elem in all_texts:
        normal_form = get_normal_form_of_single_text(elem[1])
        normalized_texts.append(normal_form)
        i += 1
        print(len(normal_form))
        print("Normalized: " + str(i))
    return normalized_texts


df = pandas.read_csv("all-reviews.csv", names=["name", "review", "label"])

my_films = ["Побег из Шоушенка", "Поймай меня, если сможешь", "Престиж"]
my_reviews = df[df.name.isin(my_films)]
train_reviews = df[~df.name.isin(my_films)]

normalized_reviews = get_normalized_texts(train_reviews.values)
vocab = get_vocab(normalized_reviews)
bag_of_words = get_bag_of_words_vectors(train_reviews.values, normalized_reviews, vocab)
print(bag_of_words)

print("Train")
regression = LogisticRegression()
regression.fit(bag_of_words, train_reviews.label)

test_normalized_reviews = get_normalized_texts(my_reviews.values)
test_bag_of_words = get_bag_of_words_vectors(my_reviews.values, test_normalized_reviews, vocab)

print("Predict")
predicted = regression.predict(test_bag_of_words)

print('Coefficients: \n', regression.coef_)
print(predicted)
print(precision_recall_fscore_support(my_reviews['label'].values, predicted))

true_positives = 0
k = 0
for prediction in predicted:
    if prediction == my_reviews.values[k][2]:
        true_positives += 1
    k += 1

print("Accuracy: {}".format(true_positives / len(predicted)))
precision_recall_fscore = precision_recall_fscore_support(my_reviews['label'].values, predicted)
print(f"Precision(-1, 0, 1) = {precision_recall_fscore[0]}")
print(f"Recall(-1, 0, 1) = {precision_recall_fscore[1]}")
print(f"Fscore(-1, 0, 1) = {precision_recall_fscore[2]}")

dict_of_negative = dict(zip(vocab, regression.coef_[0]))
dict_of_neutral = dict(zip(vocab, regression.coef_[1]))
dict_of_positive = dict(zip(vocab, regression.coef_[2]))

sorted_negative_dictionary = {k: v for k, v in sorted(dict_of_negative.items(), key=lambda item: item[1], reverse=True)}
sorted_neutral_dictionary = {k: v for k, v in sorted(dict_of_neutral.items(), key=lambda item: item[1], reverse=True)}
sorted_positive_dictionary = {k: v for k, v in sorted(dict_of_positive.items(), key=lambda item: item[1], reverse=True)}

negative = open("negatives.txt", "w", encoding="utf-8")
neutral = open("neutrals.txt", "w", encoding="utf-8")
positive = open("positives.txt", "w", encoding="utf-8")

print(sorted_negative_dictionary, file=negative)
print(sorted_neutral_dictionary, file=neutral)
print(sorted_positive_dictionary, file=positive)
