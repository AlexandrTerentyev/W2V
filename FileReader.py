import os

totalCount = 0
totalEmailCount = 27716

def enumerateMessagesDirectory(directoryPath, onReadAction):
    global totalCount
    for dirname, dirnames, filenames in os.walk(directoryPath):
        i = 1
        count = len(filenames)
        for fileName in filenames:
            print('---Enumerate ', directoryPath, ' ', i, ' of ', count,  ' current file: ', fileName, ' total: ', totalCount, ' of ', totalEmailCount)
            i+=1
            totalCount += 1
            filepath = directoryPath+"/"+fileName
            f = open(filepath, 'r')
            try:
                content = f.read()
                onReadAction(content)
            except Exception:
                print("          FAILED READ FILE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
            finally:
                f.close()

def enumerateDataSet(onReadAction):
    for i in range(1,6):
        intAsStr = str(i)
        hamPath = 'dataset/enron' + intAsStr + '/ham'
        spamPath = 'dataset/enron' + intAsStr + '/spam'
        enumerateMessagesDirectory(hamPath, onReadAction)
        enumerateMessagesDirectory(spamPath, onReadAction)

def onRead(text):
    pass

def test():
    enumerateDataSet(onRead)
    print("Total count: ", totalEmailCount)

test()
