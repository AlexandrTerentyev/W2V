import numpy as np
import tensorflow as tf
import json, codecs
import HelpFunctions as hp
import DictionaryFiller as df
import FileReader as fr

WindowSize = 2
EmbeddingVectorSize = 5
Iterations = 1
LearningRate = 0.1
wordsByIndex = {}
indexesByWords = {}
vocabularySize = 0
StudyTrainPartSize = 1000

def oneHotEncoding(unitIndex, wordCount):
    res = np.zeros(wordCount, dtype= np.uint64)
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

def createTFModel():
    Weigths1 = tf.Variable(tf.random_normal([vocabularySize, EmbeddingVectorSize]))
    bias1 = tf.Variable(tf.random_normal([EmbeddingVectorSize]))
    Weigths2 = tf.Variable(tf.random_normal([EmbeddingVectorSize, vocabularySize]))
    xTrainPlaceholder = tf.placeholder(tf.float32, shape=(None, vocabularySize))
    yTrainPlaceholder = tf.placeholder(tf.float32, shape=(None, vocabularySize))
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

def learnOnTrains(xTrain, yTrain):
    xTrainPlaceholder, yTrainPlaceholder, Weigths1, bias1, prediction, Weigths2, bias2 = createTFModel()
    session = createSession()
    lossFunction, trainStep = createLossFunction(yTrainPlaceholder, prediction)
    for i in  range (Iterations):
        print('------- Start [ ', i, ' ] iteration.')
        studyIteration(session, trainStep, lossFunction, xTrain, yTrain, xTrainPlaceholder, yTrainPlaceholder)
    return session, Weigths1, bias1, Weigths2, bias2

def studyIteration(session, trainStep, lossFunction, xIndeces, yIndeces, xTrainPlaceholder, yTrainPlaceholder):
    trainSize = len(xIndeces)
    index = 0
    # session.partial_run_setup([trainStep, lossFunction])
    while index < trainSize/100000:
        batchX = xIndeces [index: min(index+StudyTrainPartSize, trainSize)]
        batchY = yIndeces [index: min(index+StudyTrainPartSize, trainSize)]
        xTrain = [oneHotEncoding(x, vocabularySize) for x in batchX]
        yTrain = [oneHotEncoding(y, vocabularySize) for y in batchY]
        feedDictionary = {xTrainPlaceholder: xTrain, yTrainPlaceholder: yTrain}
        print('               study proccess', index / trainSize * 100, '%  Current pair part start index:', index, 'of', trainSize)
        session.run(trainStep, feed_dict=feedDictionary)
        index += StudyTrainPartSize
        # print("           loss: ", session.run(lossFunction, feed_dict=feedDictionary))



def saveResult(fileName, session, Weigths1, bias1):
    vectors = session.run(Weigths1 + bias1)
    vocabulary = {}
    for i in range (vocabularySize):
        word = wordsByIndex[str(i)]
        try:
            vocabulary[word] = vectors[i].tolist()
        except Exception:
            print('!!! Error while save result. No vector for word', word)

    jsonMap = {
        'vocabulary': vocabulary,
        'vocabularySize': vocabularySize,
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

def learn():
    print('------------ Load trains')
    xTrain, yTrain = readTrains()
    print('------------ X and Y trains prepared. Gonna start study...')
    session, Weigths1, bias1, Weigths2, bias2 = learnOnTrains(xTrain, yTrain)
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
    learn()
    # learnOnListOfTexts(['test text first', 'test text second'])

TrainsJsonFilepath = 'trains.json'

def proccessTextAndSaveTrains():
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
    json.dump(jsonMap, codecs.open(TrainsJsonFilepath, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=0)

def readTrains():
    jsonFile = open(TrainsJsonFilepath, 'r')
    trainsJSON = json.load(jsonFile)
    xIndeces = trainsJSON['xIndeces']
    yIndces = trainsJSON ['yIndeces']
    print('------------ Did read X and Y from file. Size:', len(xIndeces), 'Gonna parse to one hot...')
    return xIndeces, yIndces

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

start()