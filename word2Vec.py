import numpy as np
# import tensorflow as tf

WindowHalfSize = 2
EmbeddingVectorSize = 5
wordsByIndex = map()
indexesByWords = map()

def addWordToVocabulariesIfNeed(word):
    if indexesByWords(word):
        return
    currentVocabularySize = np.size(wordsByIndex)
    wordsByIndex[currentVocabularySize] = word
    indexesByWords [word] = currentVocabularySize

def oneHotEncoding(unitIndex, wordCount):
    res = np.zeros(wordCount)
    res[unitIndex] = 1
    return res

def allTuplesFromSentence(sentence):
    sentenceSize = np.size(sentence)
    tuples = []
    for i, centralWord in enumerate(sentence):
        addWordToVocabulariesIfNeed(centralWord)
        for j in range(max(0, i - WindowHalfSize), i):
            tuples.append((centralWord, sentence[j]))
        for j in range(i + 1, min(sentenceSize, i + WindowHalfSize)):
            tuples.append(centralWord, sentence[j])
    return tuples


