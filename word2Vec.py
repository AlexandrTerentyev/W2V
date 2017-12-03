import numpy as np
import tensorflow as tf
import json, codecs
import HelpFunctions as hp

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

def allTuplesFromSentence(sentence):
    preparedSentence = hp.preparedWordArray(sentence)
    sentenceSize = len(preparedSentence)
    tuples = []
    for i, centralWord in enumerate(preparedSentence):
        for j in range(max(0, i - WindowSize), i):
            tuples.append((centralWord, preparedSentence[j]))
        for j in range(i + 1, min(sentenceSize, i + WindowSize)):
            tuples.append((centralWord, preparedSentence[j]))
    return tuples

def trainsFromTuple(tuples):
    xTrain = []
    yTrain = []
    for wordTuple in tuples:
        xTrain.append(oneHotEncoding(indexesByWords[wordTuple[0]], vocabularySize))
        yTrain.append(oneHotEncoding(indexesByWords[wordTuple[1]], vocabularySize))
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
        session.run(trainStep, feed_dict=feedDictionary)
        print("[", i, "] loss: ", session.run(lossFunction, feed_dict=feedDictionary))
    return session, Weigths1, bias1, Weigths2, bias2

def learnAndSaveResult(xTrain, yTrain, fileName):
    session, Weigths1, bias1, Weigths2, bias2 = learn(xTrain, yTrain)
    saveResult(fileName, session, Weigths1, bias1)
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


def test():
    testText = 'Word2vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located in close proximity to one another in the space.[1]\n\nWord2vec was created by a team of researchers led by Tomas Mikolov at Google. The algorithm has been subsequently analysed and explained by other researchers.[2][3] Embedding vectors created using the Word2vec algorithm have many advantages compared to earlier algorithms[1] such as latent semantic analysis.'
    xTrain, yTrain = trainsFromTuple(allTuplesFromSentence(testText))
    session, Weigths1, bias1, Weigths2, bias2 = learnAndSaveResult(xTrain, yTrain, 'result.json')
    vectors = session.run(Weigths1 + bias1)
    print(vectors[indexesByWords[hp.prepareWord('Word2vec')]])
    print(vectors[indexesByWords[hp.prepareWord('models')]])

test()
