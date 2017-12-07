def prepareWord(word):
    return word.lower()


def preparedWordArray(sentence):
    res = sentence
    res = res.lower()
    for i in range(0, 10):
        res = res.replace(str(i), '')
    for c in [',', '.', ':', ';', '-', '_', '+', '%', '#', '@', '*', '!', '^',
              '?', '>', '<', '=', '(', ')', '>', '[', ']', '{', '}', '|', '\\', '/', '~', '`', '\"', '\'']:
        res = res.replace(c, ' ')
    res = res.split()
    return res