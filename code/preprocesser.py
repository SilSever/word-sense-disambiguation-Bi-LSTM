import string
from typing import List, Dict, Set, Tuple

import gensim
import numpy as np
from nltk import FreqDist
from nltk.corpus import stopwords
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import config
import utils

"""
    :author Silvio Severino
"""


def compute_vocabulary(
    sentences: List[str],
    min_count: int,
    num_vocab: int = None,
    vocab: Dict[str, int] = None,
) -> Tuple[Dict[str, int], FreqDist]:
    """
    This method is used to compute the vocabulary. Firstly is computes the word frequency,
    then makes the dictionary.
    :param sentences: sentences to give vocab
    :param min_count: min number of word occurrences
    :param num_vocab: maximum number of vocab
    :param vocab: if None, it computes the training vocabulary
                    else, it computes the sense inventory
    :return: the computed vocabulary and the word frequency
    """

    freq_train, frequency = compute_most_frequent_words(sentences, min_count=min_count)

    if vocab is None:
        return (
            make_vocab(sentences, num_vocab=num_vocab, most_frequent=set(freq_train)),
            frequency,
        )
    sens_inv = make_vocab(
        freq_train, sense_inv=True, start_index=len(vocab), num_vocab=num_vocab
    )

    out_vocab = {**vocab, **sens_inv}

    return out_vocab, frequency


def vocab_from_emb(embeddings: gensim.models.word2vec.Word2Vec) -> Dict[str, int]:
    """
    :param embeddings: trained Gensim Word2Vec model
    :return: a dictionary from token to int
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for index, word in enumerate(embeddings.wv.index2word):
        vocab[word] = index + 2
    return vocab


def compute_train_vocab(
    train_x: List[str],
    tr_sens: List[str],
    pos: List[str],
    min_count: int,
    num_vocab: int,
):
    """
    This method is used to compute all the training vocabularies.
    In particular,
                    - it computes the sense embeddings vocab reading the pretrained sense embeddings
                    - it computes the training vocabulary
                    - it computes the pos vocabulary
    Finally it writes all the computed vocabularies
    :param train_x: training sentences
    :param tr_sens: sense embeddings sentences
    :param pos: pos sentences
    :param min_count: min number of word occurrences
    :param num_vocab: maximum number of vocab
    :return: training vocabulary, sense embeddings vocabulary,
            pretrained embeddings, pos vocabulary, training frequency,
            pos frequency
    """
    sens_vocab, _ = compute_vocabulary(tr_sens, 0)
    emb = gensim.models.KeyedVectors.load_word2vec_format(
        config.SENSE_EMBEDDINGS, binary=True
    )
    emb = utils.squeeze_emb(emb, set(sens_vocab.keys()))
    emb.save_word2vec_format(config.SQUEEZED_EMB_TR, binary=True)
    emb_vocab = vocab_from_emb(emb)

    vocab, frequency = compute_vocabulary(
        train_x, min_count=min_count, num_vocab=num_vocab
    )

    vocab_pos, frequency_pos = compute_vocabulary(pos, min_count=0)

    utils.write_vocabulary(config.SENSE_VOCAB, emb_vocab)
    utils.write_vocabulary(config.VOCAB_TR, vocab)
    utils.write_vocabulary(config.POS_VOCAB_TR, vocab_pos)

    return vocab, emb_vocab, emb, vocab_pos, frequency, frequency_pos


def make_vocab(
    sentences: List[str],
    sense_inv: bool = False,
    start_index: int = 0,
    num_vocab: int = None,
    most_frequent: Set[str] = None,
) -> Dict[str, int]:
    """
    This method compute an only one vocabulary
    :param sentences: sentences to give vocabs
    :param sense_inv: if True, it computes a sense inventory
                        False, it computes a training vocabulary
    :param start_index: index from start the vocabulary
    :param num_vocab: maximum number of words
    :param most_frequent: a list of the most frequent words
    :return: computed vocabulary
    """

    vocab = {"<PAD>": 0, "<UNK>": 1} if not sense_inv else {}

    for sentence in sentences:
        for word in sentence.split():
            if (
                word not in vocab
                and _check_num_vocab(num_vocab, vocab)
                and _check_frequent_word(word, most_frequent)
            ):
                vocab[word] = len(vocab) + start_index
    return vocab


def _check_num_vocab(num_vocab: int, vocab: Dict[str, int]) -> bool:
    """
    This method checks whether the maximum number is reached
    :param num_vocab: maximum number of words
    :param vocab: vocabulary
    :return: True if not or the num_vocab is None
            False otherwise
    """
    return True if num_vocab is None else num_vocab != len(vocab)


def _check_frequent_word(word: str, frequent_words: Set[str]) -> bool:
    """
    This method checks whether a word is in frequent_words
    :param word: word to check
    :param frequent_words: a set of frequent words
    :return: True if so or frequent_word is None
            False otherwise
    """
    return True if frequent_words is None else word in frequent_words


def text2id_corpus(corpus: List[List[str]], vocab: Dict[str, int]) -> np.ndarray:
    """
    A simple method to handle text2id conversion
    :param corpus: corpus to convert
    :param vocab: vocab used for the conversion
    :return: a numpy array having the converted sentences
    """
    for sentences in corpus:
        yield text2id(sentences, vocab)


def text2id(sentences: List[str], vocab: Dict[str, int]) -> np.ndarray:
    """
    This method is used to tokenize and to map the word using a vocabulary
    :param sentences: sentence to convert
    :param vocab: vocab used to convert
    :return: a numpy array with the sentence tokenized and mapped
    """
    return np.array(
        [
            [
                vocab[word] if vocab.get(word) else vocab.get("<UNK>")
                for word in sentence.split()
            ]
            for sentence in sentences
        ]
    )


def padding(sentences: List[List[int]], pad: int = None) -> List[List[str]]:
    """
    This method is used to pad a sentence converted with text2id
    :param sentences: converted sentence to pad
    :param pad: maximum padding length
    :return: a padded sentence
    """
    return pad_sequences(sentences, maxlen=pad, truncating="post", padding="post")


def padding_string(sentences: List[List[str]], pad: int = None) -> np.ndarray:
    """
    This method is used to pad a sentence in ELMo format
    :param sentences: sentence to pad
    :param pad: maximum padding length
    :return: a padded sentence
    """
    return np.array(
        [" ".join(sent[:pad] + (["<PAD>"] * (pad - len(sent)))) for sent in sentences]
    )


def clear_sentences(corpus: List[List[str]], rm_stop: bool = False) -> List[str]:
    """
    This method is used to handle clear sentence method
    :param corpus: corpus to clear
    :param rm_stop: if True, remove the punctuation and the stop words
                    if False, only the punctuation
    :return: a cleared corpus
    """
    for sentences in corpus:
        cleared = clear_sentence(sentences, rm_stop)
        yield cleared


def clear_sentence(sentences: List[str], rm_stop: bool = False) -> List[str]:
    """
    This method is used to clear a list of sentences
    :param sentences: sentences to clear
    :param rm_stop: if True, remove the punctuation and the stop words
                    if False, only the punctuation
    :return: a cleared list of sentences
    """
    to_remove = (
        set(string.punctuation)
        if not rm_stop
        else set(string.punctuation) | set(stopwords.words("english"))
    )
    to_remove = "".join(list(to_remove - {"_", ":", "-"}))
    return [
        " ".join(sentence.translate(str.maketrans("", "", to_remove)).split())
        for sentence in sentences
    ]


def compute_most_frequent_words(
    sentences: List[str], min_count: int
) -> Tuple[List[str], FreqDist]:
    """
    This method is used to compute the most frequent words of a list of sentences
    :param sentences: sentences list
    :param min_count: minimum number of occurrence
    :return: a Tuple having a list of most frequent words and the frequency distribution
    """
    words = [word for sent in sentences for word in sent.split()]
    frequency = FreqDist(words)

    freq_words = [word for (word, freq) in frequency.most_common() if freq >= min_count]

    return freq_words, frequency


def compute_frequent_tr_words(
    train_x: List[str], train_y: List[str], bn2domain: Dict[str, int]
) -> List[str]:
    """
    This method is used to compute the most frequent words in according to its labels
    :param train_x: sentences list
    :param train_y: labels list
    :param bn2domain: map Babelnet to a coarse-grained domain
    :return: a list of frequent words
    """
    out = []
    for i in range(len(train_y)):
        sentences_x = train_x[i].split()
        sentences_y = train_y[i].split()
        for j in range(len(sentences_y)):
            word = sentences_y[j]
            if "_" in word and bn2domain.get(word.split("_")[-1]):
                out.append(sentences_x[j])

    return out
