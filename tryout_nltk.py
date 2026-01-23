import nltk

# ##### Tokenization
# from nltk import word_tokenize, sent_tokenize
# words = "Hello, world - how are you today?"
# # tw = word_tokenize(words)
# # print(tw)

# sentences = "Workers are alienated from the product of their labour, since others own what they produce, and they have no effective control over it because they are workers. Workers are alienated from their productive activity. Employment is forced labour: it is not the satisfaction of a human need.Workers are alienated from their human nature, because the first two aspects of separation deprive their work of those specifically human qualities that distinguish it from the activity of other animals. The worker is alienated from other workers. Instead of truly human relations between people, relations are governed by peoples’ roles as agents in the economic process of capital accumulation."
# # ts = sent_tokenize(sentences)
# # print(ts)

# ##### Stemming
# from nltk.stem import PorterStemmer
# stemmer = PorterStemmer()
# r_words = [
#     "relate",
#     "relation",
#     "relator",
#     "related",
#     "relating",
#     "relational",
#     "relatively",
#     "relata",
#     "relationality",
#     "relationship",
# ]
# # for w in r_words:
# # print(stemmer.stem(w))

# ##### POS tagging
# from nltk.corpus import state_union
# from nltk.tokenize import PunktSentenceTokenizer
# train_text = state_union.raw("2005-GWBush.txt")
# sample_text = state_union.raw("2006-GWBush.txt")

# custom_tokenizer = PunktSentenceTokenizer(train_text)
# tokenized = custom_tokenizer.tokenize(sample_text)


# def tag_em():
#     tagged = []
#     for i in tokenized:
#         words = nltk.word_tokenize(i)
#         tagged.append(nltk.pos_tag(words))

#     # print(help(str))
#     print(inspect.getsource(run))
#     print(type(tagged))
#     print(len(tagged[0:10]))
#     print(tagged[0:10])


# # tag_em()

# ##### Chunking (keeping entities together)
# # from nltk import Tree
# from nltk.draw.util import CanvasFrame
# from nltk.draw import TreeWidget


# def chunk_em():
#     tagged = []
#     for i in tokenized:
#         words = nltk.word_tokenize(i)
#         tagged = nltk.pos_tag(words)
#         chunkGram = r"""found: {<RB[RS]?>*<VB.?>*<NNP>+<NN>?} """
#         chunkParser = nltk.RegexpParser(chunkGram)
#         chunked = chunkParser.parse(tagged)
#         print(chunked)
#         # chunked.draw()  # Opens tree in new window.
#         return chunked


# # result = chunk_em()  # :: <class 'nltk.tree.tree.Tree'>
# # cf = CanvasFrame()
# # # t = Tree.fromstring("(S (NP I) (VP (V saw) (NP him)))")
# # # tc = TreeWidget(cf.canvas(), t)
# # tc = TreeWidget(cf.canvas(), result)
# # cf.add_widget(tc, 10, 10)  # (10,10) offsets
# # cf.print_to_file("chunked.ps")

# ##### Chinking (chunk everything except this chink)
# def chink_em():
#     tagged = []
#     for i in tokenized[5:]:
#         words = nltk.word_tokenize(i)
#         tagged = nltk.pos_tag(words)
#         chunkGram = r"""found: {<.*>+}
#                                }<VB.?|IN|DT|TO>+{ """
#         chunkParser = nltk.RegexpParser(chunkGram)
#         chunked = chunkParser.parse(tagged)
#         print(chunked)


# # chink_em()

# ##### Named Entity Recognition (keeping entities together)
# def ner_em():
#     tagged = []
#     for i in tokenized[50:]:
#         words = nltk.word_tokenize(i)
#         tagged = nltk.pos_tag(words)

#     named_ent = nltk.ne_chunk(tagged, binary=True)
#     named_ent.draw()


# # ner_em()

# ##### Lemmatizing (stemming but result is a whole word)
# from nltk.stem import WordNetLemmatizer

# lemmatizer = WordNetLemmatizer()


# def run_lemmatizer():
#     print(lemmatizer.lemmatize("cats"))
#     print(lemmatizer.lemmatize("cacti"))
#     print(lemmatizer.lemmatize("geese"))
#     print(lemmatizer.lemmatize("rocks"))
#     print(lemmatizer.lemmatize("python"))
#     print(lemmatizer.lemmatize("better", pos="a"))
#     print(lemmatizer.lemmatize("best", pos="a"))
#     print(lemmatizer.lemmatize("run", "v"))


# # run_lemmatizer()

# ##### Corpora
# # /Users/zach/.local/lib/python3.8/site-packages/nltk
# from nltk.corpus import gutenberg

# sample = gutenberg.raw("bible-kjv.txt")
# tokenized = sent_tokenize(sample)
# # print(tokenized[5:15])

# ##### WordNet
# from nltk.corpus import wordnet

# syns = wordnet.synsets("program")
# # print(syns)
# # print(syns[0].lemmas()[0].name()) # word
# # print(syns[0].definition()) # definition
# # print(syns[0].examples()) # examples


# def run_syns_ants(word):
#     synonyms = []
#     antonyms = []
#     for syn in wordnet.synsets(word):
#         for l in syn.lemmas():
#             # print('l:', l)
#             synonyms.append(l.name())
#             if l.antonyms():
#                 antonyms.append(l.antonyms()[0].name())

#     print({"synonyms": set(synonyms), "antonyms": set(antonyms)})


# # run_syns_ants("good")


# def run_similarities():
#     w1 = wordnet.synset("ship.n.01")
#     w2 = wordnet.synset("boat.n.01")
#     print(w1.wup_similarity(w2))  # 0.9090909090909091

#     w3 = wordnet.synset("car.n.01")
#     print(w1.wup_similarity(w3))  # 0.6956521739130435

#     w4 = wordnet.synset("cactus.n.01")
#     print(w1.wup_similarity(w4))  # 0.38095238095238093


# # run_similarities()

# ##### Text classification
from nltk.corpus import movie_reviews


def get_docs():
    return [
        (list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]


def run_classification():
    # documents = get_docs()  # Not used in this function
    #     # print(documents[0])
    #     # randomized = random.shuffle(documents)
    #     # print(randomized[0])

    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)
    #     # print(all_words.most_common(50))
    #     # print(all_words["stupid"]) # 253

    word_features = list(all_words.keys())[:3000]
    return word_features


classified = run_classification()
print(classified)  # => [str]

##### Words as features for learning.
# from timeit import default_timer as timer

# start = timer()
# # ...
# end = timer()
# print(end - start)


# def find_features(document):
#     words = set(document)
#     word_features = run_classification()
#     features = {}
#     for w in word_features:
#         features[w] = w in words
#     return features


# print(find_features(movie_reviews.words('neg/cv000_29416.txt'))) #=> dict :: {string: boolean}

# documents = get_docs()
# feature_sets = [(find_features(review), category) for (review, category) in documents]
# print(feature_sets) #=> hangs ... no output
######
# print(documents)
# print(type(documents)) #=> list :: [([str], str)]
# print(type(documents[0])) #=> tuple
# print(type(documents[-1])) #=> tuple
# print(len(documents)) #=> 2000


# def create_feature_sets(n):
#     documents = get_docs()
#     result = []
#     # start = timer()
#     first_n = documents[:n]
#     for (review, category) in first_n:
#         features = find_features(review)
#         # print(features)
#         result.append((features, category))
#         print("iteration done")

#     end = timer()
#     # print(end - start)
#     print("loop done")
#     return result


# create_feature_sets(50)


# ##### Naive Bayes
# # training_set = feature_sets[:1900]
# # testing_set = feature_sets[1900:]
# # feature_sets = create_feature_sets(20)
# # training_set = feature_sets[:10]
# # testing_set = feature_sets[10:]
# feature_sets = create_feature_sets(200)
# training_set = feature_sets[:100]
# testing_set = feature_sets[100:]

# # posterior = (prior occurrences * likelihood) / evidence

# # classifier = nltk.NaiveBayesClassifier.train(training_set)
# # print("Naive Bayes accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)
# # classifier.show_most_informative_features(15)

# ##### Save Classifier with Pickle.
# import pickle

# classifier = nltk.NaiveBayesClassifier.train(training_set)

# # save_classifier = open("naive_bayes.pickle", "wb")
# # pickle.dump(classifier, save_classifier)
# # save_classifier.close()

# classifier_f = open("naive_bayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()

# print(
#     "Original Naive Bayes accuracy percent: ",
#     (nltk.classify.accuracy(classifier, testing_set)) * 100,
# )
# classifier.show_most_informative_features(15)

# ##### Scikit-learn incorporation - NLP.
# from nltk.classify.scikitlearn import SklearnClassifier
# from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
# from sklearn.linear_model import LogisticsRegression, SGDClassifier
# from sklearn.svm import SVC, LinearSVC, NuSVC

# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# print(
#     "MNB_classifer accuracy percent: ",
#     (nltk.classify.accuracy(classifier, testing_set)) * 100,
# )

# # GaussianNB_classifier = SklearnClassifier(GaussianNB())
# # GaussianNB_classifier.train(training_set)
# # print(
# #     "GaussianNB_classifier accuracy percent: ",
# #     (nltk.classify.accuracy(GaussianNB_classifier, testing_set)) * 100,
# # )

# BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
# BernoulliNB_classifier.train(training_set)
# print(
#     "BernoulliNB_classifier accuracy percent: ",
#     (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100,
# )
