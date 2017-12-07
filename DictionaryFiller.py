import FileReader as fr
import HelpFunctions as hp
import json, codecs

wordsByIndex = {}
indexesByWords = {}
filteredWordsByIndex = {}
filteredIndexesByWords = {}
WordsByIndexesKey = 'wordsByIndexes'
VocabularySizeKey = 'vocabularySize'
IndexesByWordsKey = 'indexesByWords'
CountTreshold = 10
SortedWords = []

def vocabularySize():
    return len(wordsByIndex)

def filteredVocabularySize():
    return len(filteredWordsByIndex)

def addWordToVocabulariesIfNeed(word):
    if word in indexesByWords.keys():
        dict = indexesByWords [word]
        dict ['count'] = dict ['count'] + 1
        indexesByWords[word] = dict
        return
    currentVocabularySize = vocabularySize()
    wordsByIndex[currentVocabularySize] = word
    indexesByWords [word] = currentVocabularySize
    indexesByWords [word] = {
        'index': currentVocabularySize,
        'count': 1
    }

def proccessSentence(sentence):
    preparedSentence = hp.preparedWordArray(sentence)
    for i, centralWord in enumerate(preparedSentence):
        addWordToVocabulariesIfNeed(centralWord)

def filter():
    index = 0
    for word in indexesByWords.keys():
        if ['a', 'the', 'of', 'in', 'for', 'on', 'to', 'with', 'from', 'at', 'an', 'enron'].__contains__(word):
            continue
        info = indexesByWords[word]
        wordCount = info['count']
        if wordCount > CountTreshold:
            currentSortedWordsLen = len(SortedWords)
            newInfo = {'word': word, 'count': wordCount }
            for i in range(0, currentSortedWordsLen):
                if SortedWords[i]['count'] <= wordCount:
                    SortedWords.insert(i, newInfo)
                    break
                elif i == currentSortedWordsLen - 1:
                    SortedWords.append(newInfo)
            if currentSortedWordsLen == 0:
                SortedWords.append(newInfo)
            filteredIndexesByWords[word] = index
            filteredWordsByIndex[index] = word
            index += 1


def save(filePath):
    jsonMap = {
        VocabularySizeKey: filteredVocabularySize(),
        WordsByIndexesKey: filteredWordsByIndex,
        IndexesByWordsKey: filteredIndexesByWords
    }
    json.dump(jsonMap, codecs.open(filePath, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

def saveSortedWords(filePath):
    json.dump(SortedWords, codecs.open(filePath, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

def fillAndSave(filePath):
    fr.enumerateDataSet(proccessSentence)
    filter()
    saveSortedWords('sortedWords.json')
    save(filePath)

# fillAndSave('vocabulary.json')
