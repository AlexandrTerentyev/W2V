def prepareWord(word):
    return word.lower()


def preparedWordArray(sentence):
    res = sentence
    res = res.lower()
    res = res.replace('.', ' ')
    res = res.replace(',', ' ')
    for i in range(0, 10):
        res = res.replace(str(i), '')
    res = res.split()
    return res