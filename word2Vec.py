import math
import numpy as np
import tensorflow as tf
import json, codecs
import time
import HelpFunctions as hp
import DictionaryFiller as df
import FileReader as fr

WindowSize = 2
EmbeddingVectorSize = 8
Iterations = 10000
LearningRate = 0.1
wordsByIndex = {}
indexesByWords = {}
vocabularySize = 0
BatchSize = 64
PairsCount = 0
trainedWords = set()
LossComputingPeriod = 1
NoiseNum = 1

def currentTrainSize():
    trainSize = 256
    # trainSize = PairsCount / 100
    trainSize = int(trainSize)
    return trainSize

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
        print('          Word |', x, '| or |', y, '| is not in vocabulary')

def createTFModel():
    Weigths = tf.Variable(
        tf.truncated_normal([vocabularySize, EmbeddingVectorSize],
                            stddev=1.0 / math.sqrt(EmbeddingVectorSize))
    )
    bias = tf.Variable(tf.zeros([vocabularySize]))
    xTrainPlaceholder = tf.placeholder(tf.int32, shape=[BatchSize])
    yTrainPlaceholder = tf.placeholder(tf.int32, shape=[BatchSize, 1])
    return xTrainPlaceholder, yTrainPlaceholder, Weigths, bias

def createSession():
    session = tf.Session(config= tf.ConfigProto(log_device_placement=True))
    initializer = tf.global_variables_initializer()
    session.run(initializer)
    return session

def createLossFunction(xTrainPlaceholder, weights, bias, yTrainPlaceholder):
    embeddings = tf.Variable(
        tf.random_uniform([vocabularySize, EmbeddingVectorSize], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, xTrainPlaceholder)
    # W1 = tf.Variable(
    #     tf.random_normal([BatchSize, EmbeddingVectorSize],
    #                         stddev=1.0 / math.sqrt(EmbeddingVectorSize))
    # )
    # bias1 = tf.Variable(tf.random_normal([EmbeddingVectorSize]))
    # hiddenRepresentation = tf.add(tf.matmul(xTrainPlaceholder, W1), bias1)
    lossFunction = tf.reduce_mean(
        tf.nn.nce_loss(weights=weights,
                 biases=bias,
                 labels=yTrainPlaceholder,
                 inputs=embed,
                 num_sampled=NoiseNum,
                 num_classes=vocabularySize))
    trainStep = tf.train.GradientDescentOptimizer(LearningRate).minimize(lossFunction)
    return lossFunction, trainStep, embeddings

def learnOnTrains(xTrain, yTrain):
    xTrainPlaceholder, yTrainPlaceholder, Weigths, bias = createTFModel()
    lossFunction, trainStep, embeddings = createLossFunction(xTrainPlaceholder, Weigths, bias, yTrainPlaceholder)
    session = createSession()
    lossList = []
    for i in  range (Iterations):
        print('------- Start [ ', i, ' ] iteration.')
        studyIteration(i,session, trainStep, lossFunction, lossList, xTrain, yTrain, xTrainPlaceholder, yTrainPlaceholder)
    saveLoss(lossList)
    return session, Weigths, bias, embeddings

def studyIteration(iteration, session, trainStep, lossFunction, lossList, xIndeces, yIndeces, xTrainPlaceholder, yTrainPlaceholder):
    trainSize = currentTrainSize()
    index = 0
    numOfOperations = 1
    lastProgress = 0
    # session.partial_run_setup([trainStep, lossFunction])
    stepTimeSum = 0
    while index < trainSize:
        batchEnd = min(index + BatchSize, trainSize)
        batchX = np.ndarray(shape=(BatchSize), dtype=np.int32)
        batchY = np.zeros([BatchSize, 1])
        j = 0
        for i in range(index, batchEnd):
            batchY[j, 0] = yIndeces[i]
            batchX[j] = xIndeces[i]
            j += 1
        for x in batchX:
            trainedWords.add(x)
        feedDictionary = {xTrainPlaceholder: batchX, yTrainPlaceholder: batchY}
        progress = (batchEnd + iteration+trainSize) / trainSize * 100 / Iterations
        startIteration =time.time()
        session.run(trainStep, feed_dict=feedDictionary)
        endIteration = time.time()
        stepProgress = progress - lastProgress
        stepTime = endIteration - startIteration
        stepTimeSum += stepTime
        timeRemain = 100 / stepProgress * (stepTimeSum/numOfOperations)
        finishTimestamp = time.time()+timeRemain
        finishTime = time.strftime("%D %H:%M", time.localtime(finishTimestamp))
        currentTime = time.strftime("%D %H:%M:%S", time.localtime(time.time()))
        print('   ', currentTime, '    study proccess', progress, '%  Current pair part start index:', index, 'of', trainSize,
              'Finish time:', finishTime)
        index += BatchSize
        lastProgress = progress
        if numOfOperations % LossComputingPeriod == 0:
            loss = session.run(lossFunction, feed_dict=feedDictionary)
            lossList.append(loss)
            print("loss: ", loss)
        numOfOperations += 1


def saveLoss(lossList):
    fileName = "LOSS_" + time.strftime("%d_%m_%Y_%H_%M_%S", time.localtime(time.time())) + '.json'
    jsonMap = {
        'loss': [float(x) for x in lossList]
    }
    json.dump(jsonMap, codecs.open(fileName, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

def saveResult(fileName, embeddings):
    vectors = embeddings
    vocabulary = {}
    for i in range (vocabularySize):
        if not trainedWords.__contains__(i):
            continue
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
    session, Weigths1, bias1, embeddings = learnOnTrains(xTrain, yTrain)
    print('------------ Study completed. Gonna save result...')
    saveResult('result.json', embeddings.eval(session))
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
    global PairsCount
    jsonFile = open(TrainsJsonFilepath, 'r')
    trainsJSON = json.load(jsonFile)
    xIndeces = trainsJSON['xIndeces']
    yIndces = trainsJSON ['yIndeces']
    PairsCount = len(xIndeces)
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
# proccessTextAndSaveTrains()