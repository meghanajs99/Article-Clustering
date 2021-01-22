def process(para, filename):
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import wordnet as wn
    nltk.download('wordnet')
    from nltk.corpus import stopwords
    from collections import defaultdict
    stop_words = set(stopwords.words('english'))
    sents = re.findall("[\w']+", para)
    tokens = []
    for x in sents:
        if len(x) > 2 and not x.isdigit():
            tokens.append(x)
    filtered_sentence = [w for w in tokens if not w in stop_words]
    for w in tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in filtered_sentence]
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    nouns = set()
    for word, pos in nltk.pos_tag(stemmed):
        if pos in ['NN', "NNP"]:
            nouns.add(word)
        word = []
    for x in nouns:
        if x not in word:
            word.append(x)
    return word
