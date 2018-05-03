# python3 homework3.py train.txt dev.txt
# c=1,k=1: Precision:0.9302325581395349 Recall:0.5970149253731343 F-Score:0.7272727272727273 Accuracy:0.9453551912568307
# python3 homework3.py train.txt test.txt
# c=1,k=1: Precision:0.9555555555555556 Recall:0.6825396825396826 F-Score:0.7962962962962963 Accuracy:0.9597806215722121
# c=0.0001,k=6: Precision:0.8627450980392157 Recall:0.6984126984126984 F-Score:0.7719298245614035 Accuracy:0.9524680073126143

import sys
import string
import math


# function to check if a list of words are present in the vocabulary
def check_vocabulary(word_list, vocab):
    valid_word_list = []
    for val in word_list:
        if val in vocab:
            valid_word_list.append(val)
    return valid_word_list


# function to extract the words from the message text passed
def extract_words(text):
    # convert text to lower case
    lower_case_text = text.lower()

    # remove punctuation from the text message
    table = str.maketrans({key: None for key in string.punctuation})
    no_punctuation_text = lower_case_text.translate(table)
    word_list = no_punctuation_text.split(" ")
    return word_list


# Function to calculate the prior probability
def get_prior_prob(training_filename):
    word_label_map = {}
    label_count_map = {}
    prior_prob_map = {}
    with open(training_filename, "r") as file:
        for line in file:
            train_file_label, train_file_line = line.split("\t")
            if train_file_label not in word_label_map:
                word_label_map[train_file_label] = [train_file_line.rstrip()]
            else:
                word_label_map[train_file_label].append(train_file_line.rstrip())

    file.close()

    word_label_frequency_list = word_label_map.keys()

    for item in word_label_frequency_list:
        message_list = word_label_map[item]
        label_count_map[item] = len(message_list)

    label_count_list = label_count_map.keys()

    total = sum(label_count_map.values())

    for label in label_count_list:
        label_count = label_count_map[label]
        prior_prob_map[label] = label_count/total

    return prior_prob_map


# Function to get the conditional probability of a word given label
def get_word_given_label_prob(self, training_filename):
    word_label_map = {}
    word_count_per_label_map = {}
    word_given_label_prob_map = {}

    # smoothing parameter
    c = 0.0001
    vocab_size = len(self.attribute_types)

    # open the training file
    with open(training_filename, "r") as file:
        for line in file:
            train_file_label, train_file_line = line.split("\t")
            pre_processed_word_list = extract_words(train_file_line.rstrip())

            # check if the word are present in the vocabulary
            new_list = check_vocabulary(pre_processed_word_list, self.attribute_types)

            # get the word given label count
            for word in new_list:
                count = word_label_map.get((word, train_file_label), 0)
                word_label_map[(word, train_file_label)] = count + 1

    file.close()

    word_label_frequency_list = word_label_map.keys()

    # get the word count per label
    for item in word_label_frequency_list:
        word, label = item[0], item[1]
        count = word_count_per_label_map.get(label, 0)
        word_count_per_label_map[label] = count + 1

    for word in self.attribute_types:
        for lbl in self.label_prior:

            # check if the word, label tuple is in the map and perform smoothing
            if (word, lbl) in word_label_map:
                count_word_given_label = word_label_map[(word, lbl)] + c
            else:
                count_word_given_label = c

            # compute the conditional probability
            word_count_per_label = word_count_per_label_map[lbl] + (c * vocab_size)
            prob_word_given_label = count_word_given_label / word_count_per_label

            word_given_label_prob_map[(word, lbl)] = prob_word_given_label

    # return the conditional probability map
    return word_given_label_prob_map


class NbClassifier(object):

    def __init__(self, training_filename, stopword_file="stopwords_mini.txt"):
        self.attribute_types = set()
        self.label_prior = {}    
        self.word_given_label = {}   

        # set the cut-off parameter k
        self.collect_attribute_types(training_filename, 6)

        # read the stopwords file
        stopwords_list = []
        with open(stopword_file, "r") as file:
            for line in file:
                stopwords_list.append(line.rstrip())

        file.close()

        # remove stopwords from the vocabulary
        self.attribute_types = list(set(self.attribute_types).difference(stopwords_list))

        self.train(training_filename)

    def collect_attribute_types(self, training_filename, k):
        frequency = {}
        attribute_types = []

        # open the train file
        with open(training_filename, "r") as file:
            for line in file:
                # split the line by tab to extract the text message
                train_line = line.split("\t")
                word_in_text = train_line[1].rstrip().split(" ")

                # get the word frequency
                for word in word_in_text:
                    count = frequency.get(word, 0)
                    frequency[word] = count + 1

        frequency_list = frequency.keys()

        # remove words that occur less than k times
        for words in frequency_list:
            if frequency[words] > k:
                attribute_types.append(words)

        file.close()

        self.attribute_types = attribute_types

    def train(self, training_filename):

        # get the prior probability
        prior_prob_map = get_prior_prob(training_filename)

        self.label_prior = prior_prob_map

        # get the conditional probability of the word given label
        word_given_label_prob_map = get_word_given_label_prob(self, training_filename)

        self.word_given_label = word_given_label_prob_map

    def predict(self, text):
        # split the text message by space
        word_list = text.split(" ")
        prob_prediction_map = {}

        # use the prior and conditional probabilities to predict the probability
        # that the text message is classified as 'spam' or 'ham'
        label_prior = self.label_prior
        word_given_label = self.word_given_label

        prior_keys = label_prior.keys()

        # compute the log probabilities
        for key in prior_keys:
            prob_list = []
            prior_prob_label = math.log(label_prior[key])
            prob_list.append(prior_prob_label)
            for word in word_list:
                if (word, key) in word_given_label:
                    cond_prob = math.log(word_given_label[(word, key)])
                else:
                    cond_prob = 0
                prob_list.append(cond_prob)
            prob_prediction_map[key] = sum(prob_list)

        return prob_prediction_map

    def evaluate(self, test_filename):
        test_map = {}

        # open the test file
        with open(test_filename, "r") as file:
            for line in file:
                # split each line of the file by tab
                test_file_label, test_file_line = line.split("\t")

                # call the predict function
                prob_prediction_map = self.predict(test_file_line)

                # classify the text as spam or ham depending on the probability computed
                if prob_prediction_map["spam"] > prob_prediction_map["ham"]:
                    classify_label = "spam"
                else:
                    classify_label = "ham"

                test_map[test_file_line] = (test_file_label, classify_label, prob_prediction_map["ham"],
                                            prob_prediction_map["spam"])

        # find the number of true positives, true negatives,
        # false positives and false negatives
        test_keys = test_map.keys()
        tp, tn, fn, fp = 0, 0, 0, 0

        for test_key in test_keys:
            test_value = test_map[test_key]

            if (test_value[0] == "spam") and (test_value[1] == "spam"):
                tp = tp + 1
            if (test_value[0] == "ham") and (test_value[1] == "ham"):
                tn = tn + 1

            if (test_value[0] != test_value[1]) and (test_value[1] == "spam"):
                fp = fp + 1
            elif (test_value[0] != test_value[1]) and (test_value[1] == "ham"):
                fn = fn + 1

        # calculate the precision, recall, f-score and accuracy
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fscore = (2 * precision * recall) / (precision + recall)

        # return the values for precision, recall, f-score and accuracy
        return precision, recall, fscore, accuracy


def print_result(result):
    print("Precision:{} Recall:{} F-Score:{} Accuracy:{}".format(*result))


if __name__ == "__main__":
    
    classifier = NbClassifier(sys.argv[1])
    result = classifier.evaluate(sys.argv[2])
    print_result(result)
