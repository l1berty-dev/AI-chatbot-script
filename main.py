import time
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from config import is_training, understand_limit
from recognize import get_data, triggers, get_data_time, triggers_search
import pyttsx3
from functions import check_time, web_browser

stemmer = LancasterStemmer()
voice = pyttsx3.init()
volume = voice.getProperty('volume')
voice.setProperty('volume', 1.0)
# voice.setProperty('voice', voice_type[1].id)

with open("intents.json", encoding='utf-8') as file:
    data = json.load(file)

if is_training:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
else:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

if is_training:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    model.load("model.tflearn")
else:
    model.load("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat(voice):
    while True:
        inp = get_data()
        trg = triggers.intersection(inp.split())
        trg_sr = triggers_search.intersection(inp.split())
        if not trg:
            continue
        inp = inp.replace(list(trg)[0], '')

        while True:
            print(inp)
            if trg_sr:
                print(trg_sr)
                inp = inp.replace(list(trg_sr)[0], '')
                web_browser(inp)
                break
            results = model.predict([bag_of_words(inp, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]
            if results[results_index] > understand_limit:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        if tg['tag'] == "check_time_intent":
                            voice.say(check_time())
                            voice.runAndWait()
                            inp = get_data_time()
                            if inp == "":
                                text_exist = False
                                break
                            else:
                                text_exist = True
                                break

                        responses = tg['responses']
                        voice.say(random.choice(responses))
                        voice.runAndWait()
                        inp = get_data_time()
                        if inp == "":
                            text_exist = False
                            break
                        else:
                            text_exist = True
                            break
                if text_exist:
                    continue
                else:
                    break


            else:
                voice.say("Простите, не понял")
                voice.runAndWait()
                inp = get_data_time()
                if inp == "":
                    break
                else:
                    continue


chat(voice)
