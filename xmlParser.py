import xml.etree.ElementTree as ET

ArticlesCount = 10000

def parse(onReadAction):
    i = 0
    for event, elem in ET.iterparse('enwiki-20170820-pages-articles.xml', events=('start', 'end')):
        text = elem.text
        if 'text' in elem.tag and text is not None and '#REDIRECT' not in text:
            onReadAction(elem.text)
            print('Proccess', i, 'of', ArticlesCount, 'articles.  ', i/ArticlesCount * 100,'%')
            i+=1
            if i > ArticlesCount:
                break
