import FileReader as fr
import HelpFunctions as hp
import json, codecs

wordsByIndex = {}
indexesByWords = {}

def vocabularySize():
    return len(wordsByIndex)

def addWordToVocabulariesIfNeed(word):
    if word in indexesByWords.keys():
        # dict = indexesByWords [word]
        # dict ['count'] = dict ['count'] + 1
        # indexesByWords[word] = dict
        return
    currentVocabularySize = vocabularySize()
    wordsByIndex[currentVocabularySize] = word
    indexesByWords [word] = currentVocabularySize
    # indexesByWords [word] = {
    #     'index': currentVocabularySize,
    #     'count': 1
    # }

def proccessSentence(sentence):
    preparedSentence = hp.preparedWordArray(sentence)
    for i, centralWord in enumerate(preparedSentence):
        addWordToVocabulariesIfNeed(centralWord)

def save(filePath):
    jsonMap = {
        'vocabularySize': vocabularySize(),
        'wordsByIndexes': wordsByIndex,
        'indexesByWords': indexesByWords
    }
    json.dump(jsonMap, codecs.open(filePath, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

def fillAndSave(filePath):
    fr.enumerateDataSet(proccessSentence)
    save(filePath)

fillAndSave('vocabulary.json')
