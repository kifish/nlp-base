import logging
#https://gist.github.com/danoneata/3a42abbd3aece67ac07d15e988d88ae1
#实际上还是经常会报错,截止2018-05-13,genism的doc2vec不完全支持online-learning-doc2vec
from gensim.models.doc2vec import (
    Doc2Vec,
    TaggedDocument,
)

logging.basicConfig(
    format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
    level=logging.DEBUG,
)


def to_str(d):
    return ", ".join(d.keys())


SENTS = [
    "anecdotal using a personal experience or an isolated example instead of a sound argument or compelling evidence",
    "plausible thinking that just because something is plausible means that it is true",
    "occam razor is used as a heuristic technique discovery tool to guide scientists in the development of theoretical models rather than as an arbiter between published models",
    "karl popper argues that a preference for simple theories need not appeal to practical or aesthetic considerations",
    "the successful prediction of a stock future price could yield significant profit",
]

SENTS = [s.split() for s in SENTS]


def main():
    sentences_1 = [
        TaggedDocument(SENTS[0], tags=['SENT_0']),
        TaggedDocument(SENTS[1], tags=['SENT_0']),
        TaggedDocument(SENTS[2], tags=['SENT_1']),
    ]

    sentences_2 = [
        TaggedDocument(SENTS[3], tags=['SENT_1']),
        TaggedDocument(SENTS[4], tags=['SENT_2']),
    ]

    model = Doc2Vec(min_count=1, workers=1)

    model.build_vocab(sentences_1)
    model.train(sentences_1,total_examples=model.corpus_count,epochs=model.epochs)

    print("-- Base model")
    #print("Vocabulary:", to_str(model.wv.vocab.keys()))
    print("Vocabulary:", model.wv.vocab.keys())
    print("Tags:", to_str(model.docvecs.doctags))

    model.build_vocab(sentences_2, update=True)
    model.train(sentences_2,total_examples=model.corpus_count,epochs=model.epochs)

    print("-- Updated model")
    #print("Vocabulary:", to_str(model.wv.vocab.keys()))
    print("Vocabulary:", model.wv.vocab.keys())
    print("Tags:", to_str(model.docvecs.doctags))


if __name__ == '__main__':
    main()
