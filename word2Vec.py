import numpy as np
import tensorflow as tf
import json, codecs
import HelpFunctions as hp
import DictionaryFiller as df
import FileReader as fr

WindowSize = 2
EmbeddingVectorSize = 5
Iterations = 1000
LearningRate = 0.1
wordsByIndex = {}
indexesByWords = {}
vocabularySize = 0

def oneHotEncoding(unitIndex, wordCount):
    res = np.zeros(wordCount)
    res[unitIndex] = 1
    return res

def allTuplesFromSentences(sentences):
    tuples = []
    i = 0
    count = len(sentences)
    for sentence in sentences:
        tuples += allTuplesFromSentence(sentence)
        i += 1
        print('   ', i, 'of', count, ' sentences processed. ', (i/count * 100), '%')
    return tuples

def allTuplesFromSentence(sentence):
    preparedSentence = hp.preparedWordArray(sentence)
    sentenceSize = len(preparedSentence)
    tuples = []
    count = len(preparedSentence)
    for i, centralWord in enumerate(preparedSentence):
        print('      ', i, 'of', count, ' words processed. ', (i / count * 100), '%')
        for j in range(max(0, i - WindowSize), i):
            tuples.append((centralWord, preparedSentence[j]))
        for j in range(i + 1, min(sentenceSize, i + WindowSize)):
            tuples.append((centralWord, preparedSentence[j]))
    return tuples

def xAndYTrainsFromSentence(sentence):
    preparedSentence = hp.preparedWordArray(sentence)
    sentenceSize = len(preparedSentence)
    xTrain = []
    yTrain = []
    count = len(preparedSentence)
    for i, centralWord in enumerate(preparedSentence):
        print('      ', i, 'of', count, ' words processed. ', (i / count * 100), '%')
        for j in range(max(0, i - WindowSize), i):
            addWordsToXandYTrains(centralWord, preparedSentence[j], xTrain, yTrain)
        for j in range(i + 1, min(sentenceSize, i + WindowSize)):
            addWordsToXandYTrains(centralWord, preparedSentence[j], xTrain, yTrain)
    return xTrain, yTrain

def addWordsToXandYTrains(x, y, xTrain, yTrain):
    try:
        xTrain.append(indexesByWords[x])
        yTrain.append(indexesByWords[y])
    except:
        print('          Word', x, 'or', y, 'is not in vocabulary')



def trainsFromTuples(tuples):
    xTrain = []
    yTrain = []
    i=0
    count = len(tuples)
    for wordTuple in tuples:
        try:
            xTrain.append(oneHotEncoding(indexesByWords[wordTuple[0]], vocabularySize))
            yTrain.append(oneHotEncoding(indexesByWords[wordTuple[1]], vocabularySize))
        except Exception:
            print('          Word',[wordTuple[0]], 'or', wordTuple[1], 'is not in vocabulary' )
        print('      ', i, 'of', count, ' tuples processed to X and Y train. ', (i / count * 100), '%')
        i+=1
    return np.asarray(xTrain), np.asarray(yTrain)

def createTFModel():
    xTrainPlaceholder = tf.placeholder(tf.float32, shape=(None, vocabularySize))
    yTrainPlaceholder = tf.placeholder(tf.float32, shape=(None, vocabularySize))
    Weigths1 = tf.Variable(tf.random_normal([vocabularySize, EmbeddingVectorSize]))
    bias1 = tf.Variable(tf.random_normal([EmbeddingVectorSize]))
    Weigths2 = tf.Variable(tf.random_normal([EmbeddingVectorSize, vocabularySize]))
    hiddenRepresentation = tf.add(tf.matmul(xTrainPlaceholder, Weigths1), bias1)
    bias2 = tf.Variable(tf.random_normal([vocabularySize]))
    prediction = tf.nn.softmax(tf.add(tf.matmul(hiddenRepresentation, Weigths2), bias2))
    return xTrainPlaceholder, yTrainPlaceholder, Weigths1, bias1, prediction, Weigths2, bias2

def createSession():
    session = tf.Session()
    initializer = tf.global_variables_initializer()
    session.run(initializer)
    return session

def createLossFunction(yPlaceHolder, prediction):
    lossFunction = tf.reduce_mean(-tf.reduce_sum(yPlaceHolder * tf.log(prediction), axis=[1]))
    trainStep = tf.train.GradientDescentOptimizer(LearningRate).minimize(lossFunction)
    return lossFunction, trainStep

def learn(xTrain, yTrain):
    xTrainPlaceholder, yTrainPlaceholder, Weigths1, bias1, prediction, Weigths2, bias2 = createTFModel()
    session = createSession()
    lossFunction, trainStep = createLossFunction(yTrainPlaceholder, prediction)
    feedDictionary = {xTrainPlaceholder: xTrain, yTrainPlaceholder: yTrain}
    for i in  range (Iterations):
        print('Start [ ', i, ' ] iteration.')
        session.run(trainStep, feed_dict=feedDictionary)
        print("Iteration [", i, "] loss: ", session.run(lossFunction, feed_dict=feedDictionary))
    return session, Weigths1, bias1, Weigths2, bias2

def saveResult(fileName, session, Weigths1, bias1):
    vectors = session.run(Weigths1 + bias1)
    vocabulary = {}
    for i in range (vocabularySize()):
        vocabulary [wordsByIndex[i]] = vectors[i].tolist()
    jsonMap = {
        'vocabulary': vocabulary,
        'vocabularySize': vocabularySize(),
        'vectorSize': EmbeddingVectorSize
    }
    json.dump(jsonMap, codecs.open(fileName, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    # with open(fileName, 'w') as outfile:
    #     js.dump(jsonMap, outfile)

def loadVocabulary (filePath):
    global vocabularySize, wordsByIndex, indexesByWords
    jsonFile = open(filePath, 'r')
    vocabulary = json.load(jsonFile)
    vocabularySize = vocabulary [df.VocabularySizeKey]
    wordsByIndex = vocabulary [df.WordsByIndexesKey]
    indexesByWords = vocabulary [df.IndexesByWordsKey]

def learnOnText(text):
    print('------------ Gonna parse text list to tuples...')
    tuples = allTuplesFromSentence(text)
    print('------------ Tuple list filled. Size: ', len(tuples))
    xTrain, yTrain = trainsFromTuples(tuples)
    print('------------ X and Y trains prepared. Gonna start study...')
    session, Weigths1, bias1, Weigths2, bias2 = learn(xTrain, yTrain)
    print('------------ Study completed. Gonna save result...')
    saveResult('result.json', session, Weigths1, bias1)
    # print(vectors[indexesByWords[hp.prepareWord('Word2vec')]])
    # print(vectors[indexesByWords[hp.prepareWord('models')]])

def concatenateAllEmails():
    emails = []
    def onReadEmail(emailText):
        emails.append(emailText)
    fr.enumerateDataSet(onReadEmail)
    return emails

def start():
    print('------------ Load vocabulary...')
    loadVocabulary('vocabulary.json')
    print('------------ Read all emails file...')
    text = readAllEmailsFile()
    learnOnText(text)
    # learnOnListOfTexts(['test text first', 'test text second'])

def proccessTextAndSaveTuples():
    print('------------ Load vocabulary...')
    loadVocabulary('vocabulary.json')
    print('------------ Read all emails file...')
    text = readAllEmailsFile()
    print('------------ Gonna parse text to trains...')
    xTrain, yTrain = xAndYTrainsFromSentence(text)
    jsonMap = {
        'xIndeces': xTrain,
        'yIndeces': yTrain,
        'size': len(xTrain)
    }
    json.dump(jsonMap, codecs.open('trains.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=0)


def readAllEmailsFile():
    f = open('allEmails.txt', 'r')
    text = f.read()
    f.close()
    return text

def saveConcatenatedEmails():
    text = ''
    for email in concatenateAllEmails():
        text += email
    f = open('allEmails.txt', 'w')
    f.write(text)
    f.close()

proccessTextAndSaveTuples()