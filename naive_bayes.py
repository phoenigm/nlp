import nltk as nltk
import pandas
import pymorphy2
import numpy


def get_normal_form(text_values):
    analyzer = pymorphy2.MorphAnalyzer()
    punctuation = [',', '.', ':', '?', '«', '»', '-', '(', ')', '!', '\'', '—', ';', '”', '...', '—', '!', '\'']
    normalized_words = []
    for text in text_values:
        tokens = nltk.word_tokenize(text)
        for token in tokens:
            if token in punctuation: continue
            normalized_words.append(analyzer.parse(token)[0].normal_form)
    return normalized_words


df = pandas.read_excel("all-reviews.xlsx", names=["name", "review", "label"])

my_films = ["Побег из Шоушенка", "Поймай меня, если сможешь", "Престиж"]
my_reviews = df[df.name.isin(my_films)]
negatives = df[df.label.eq(-1) & ~df.name.isin(my_films)]
neutrals = df[df.label.eq(0) & ~df.name.isin(my_films)]
positives = df[df.label.eq(1) & ~df.name.isin(my_films)]

my_reviews_normalized = get_normal_form(my_reviews.review.values)
negatives_normalized = get_normal_form(negatives.review.values)
neutrals_normalized = get_normal_form(neutrals.review.values)
positives_normalized = get_normal_form(positives.review.values)

unique_words = numpy.unique(negatives_normalized + neutrals_normalized + positives_normalized)
p_plus = p_minus = p_zero = 1 / 3

count_unique_words = len(unique_words)


def predict(review):
    normalized_review = get_normal_form(review)
    positive_vector = []
    negative_vector = []
    neutral_vector = []

    positive_divider = (len(positives_normalized) + count_unique_words)
    negative_divider = (len(negatives_normalized) + count_unique_words)
    neutral_divider = (len(neutrals_normalized) + count_unique_words)

    for word in normalized_review:
        count_in_positive = positives_normalized.count(word)
        positive_vector.append((count_in_positive + 1) / positive_divider)

        count_in_neutral = neutrals_normalized.count(word)
        neutral_vector.append((count_in_neutral + 1) / neutral_divider)

        count_in_negative = negatives_normalized.count(word)
        negative_vector.append((count_in_negative + 1) / negative_divider)

        positive_prediction = numpy.log(p_plus)
        neutral_prediction = numpy.log(p_zero)
        negative_prediction = numpy.log(p_minus)

        for multiplier in positive_vector:
            positive_prediction += numpy.log(multiplier)
        for multiplier in neutral_vector:
            neutral_prediction += numpy.log(multiplier)
        for multiplier in negative_vector:
            negative_prediction += numpy.log(multiplier)

    return [negative_prediction, neutral_prediction, positive_prediction]


index = 0
count_correct_predictions = 0
for my_review in my_reviews.values:
    predictions = predict(my_review[1])
    index += 1
    predicted_value = 0
    if (predictions[0] > predictions[1]) & (predictions[0] > predictions[2]):
        predicted_value = -1
    if (predictions[1] > predictions[0]) & (predictions[1] > predictions[2]):
        predicted_value = 0
    if (predictions[2] > predictions[0]) & (predictions[2] > predictions[1]):
        predicted_value = 1
    print(f"{index}, {predictions[0]:.8f}, {predictions[1]:.8f},"
          f" {predictions[2]:.8f}, {predicted_value}, {my_review[2]}")
    if f"{predicted_value}" == my_review[2]:
        count_correct_predictions += 1

print("Accuracy: {}".format(count_correct_predictions / index))
