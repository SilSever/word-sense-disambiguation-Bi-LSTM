from collections import defaultdict
from typing import Dict, List, Tuple, Set

import gensim
import numpy as np
from nltk import FreqDist
from nltk.corpus import wordnet as wn

import config

"""
    :author Silvio Severino
"""


def write_sentences_and_labels(
    sentences_path: str, file: List[Tuple[str, str, str]]
) -> None:
    """
    This method is used to write the parsed sentences, labels and pos
    :param sentences_path: path where it has to write
    :param file: file to write
    :return: None
    """
    with open(str(sentences_path), mode="w") as sentence_file:
        for sent, lab, pos in file:
            sentence_file.write(sent + "\t" + lab + "\t" + pos + "\n")


def read_sentences_and_labels(
    sentences_path: str
) -> Tuple[List[str], List[str], List[str]]:
    """
    This method is used to read the sentences, labels and pos
    :param sentences_path: path from read
    :return: sentences list, labels list, pos's list
    """
    out_x, out_y, out_pos = [], [], []

    with open(str(sentences_path), mode="r") as sentence_file:
        for sent in sentence_file:
            sent = sent.strip().split("\t")
            out_x.append(sent[0])
            out_y.append(sent[1])
            out_pos.append(sent[2])

    return out_x, out_y, out_pos


def write_vocabulary(filepath: str, vocab: Dict[str, int]) -> None:
    """
    This method is used to write a vocabulary
    :param filepath: path where it has to write
    :param vocab: vocab to write
    :return: None
    """
    with open(str(filepath), mode="w") as file:
        for key in vocab:
            file.write(key + "\t" + str(vocab[key]) + "\n")


def read_vocabulary(filepath: str) -> Dict[str, int]:
    """
    This method is used to read a vocabulary
    :param filepath: path from read
    :return: a vocabulary
    """
    vocab = defaultdict(int)
    with open(str(filepath), mode="r") as file:
        for line in file:
            line = line.strip().split("\t")
            vocab[line[0]] = int(line[1])
    return vocab


def read_map(
    filepath: str, reverse: bool = False, delimiter: str = "\t"
) -> Dict[str, str]:
    """
    This method is used to read a map of strings
    :param filepath: path where it has to read
    :param reverse: if True, it builds a map value2key
                    if False, it builds a map key2value
    :param delimiter: file delimiter
    :return: a map
    """
    out_dict = defaultdict(str)
    with open(str(filepath), mode="r") as file:
        for line in file:
            line = line.strip().split(delimiter)
            if reverse:
                out_dict[line[1]] = line[0]
            else:
                out_dict[line[0]] = line[1]
    return out_dict


def retrieve_senses(
    sentences: List[str], bn2domain: Dict[str, str], is_bn: bool = True
) -> List[str]:
    """
    This method is used to retrieve all the senses from a list of sentences
    :param sentences: list of sentences
    :param bn2domain: a map Babelnet to a coarse-grained domain
    :param is_bn: if True, it retrieves a Babelnet sense, splitting it in order to get it synset
                if False, it retrieve a coarse-grained sense
    :return: a list of senses
    """
    return [
        word
        for sentence in sentences
        for word in sentence.split()
        if bn2domain.get(word.split("_")[-1] if is_bn else word)
    ]


def input_sens_emb(sentences: List[str], bn2domain: Dict[str, str]) -> List[str]:
    """
    This method is used to compute the sense embedding sentences.
    in the format S = ['unk', ... , lemma_BabelnetSynset_i, ..., 'unk']
    :param sentences: sentence to convert
    :param bn2domain: a map Babelnet to domain used to check the right sense
    :return: a converted list
    """
    out = []
    for sentence in sentences:
        for word in sentence.split():
            if bn2domain.get(word.split("_")[-1]):
                out.append(word)
            else:
                out.append("unk")
    return out


def senses_position_from_vocab(
    lemma: str, vocab: Dict[str, int], map_domain: Dict[str, str]
) -> List[int]:
    """
    This method is used to compute the sense position from a vocab given a lemma
    :param lemma: lemma
    :param vocab: vocab from retrieve
    :param map_domain: a map used to check the right sense
    :return: a list of positions
    """
    return [vocab[elem] for elem in vocab if _check_lemma(elem, lemma, map_domain)]


def _check_lemma(elem: str, lemma: str, map_domain: Dict[str, str]) -> bool:
    """
    This method is used to check whether a lemma is the right sense
    :param elem: elem to check
    :param lemma: lemma
    :param map_domain: a map used to check the right sense
    :return: True if so,
            False otherwise
    """
    elem = elem.split("_")
    return map_domain.get(elem[-1]) and "_".join(elem[:-1]) == lemma


def reverse_vocab(vocab: dict) -> dict:
    """
    This method is used to reverse a vocabulary
    :param vocab: vocab to reverse
    :return: reversed vocab
    """
    return {v: k for k, v in vocab.items()}


def most_frequent_sense(
    lemma: str,
    pos: str,
    wn2bn: Dict[str, str],
    bn2domain: Dict[str, str] = None,
    is_bn=False,
) -> str:
    """
    This method is used to compute the most frequent sense given a lemma, exploiting Wordnet
    :param lemma: lemma
    :param pos: pos of lemma
    :param wn2bn: a map Wordnet to Babelnet
    :param bn2domain: a map Babelnet to a domain
    :param is_bn: if True, it returns a Babelnet synset
                if False, it return a coarse-grained synset
    :return: the most frequent sense
    """
    mfs = wn.synsets(lemma, pos=pos)[0]
    syn = "wn:" + str(mfs.offset()).zfill(8) + mfs.pos()

    return _get_frequent_sense(syn, wn2bn, bn2domain, is_bn)


def convert_goldkey2domain(
    input_path: str, output_path: str, bn2domain: Dict[str, str], is_bn: bool = True
) -> None:
    """
    This method is used to convert a goldkey map
    :param input_path: path of goldkey to convert
    :param output_path: path where it writes
    :param bn2domain: a map Babelnet 2 domain
    :param is_bn: if True, it converts in Babelnet format
                if False, it converts in coarse-grained format
    :return: None
    """

    wn2bn = read_map(config.BABELNET2WORDNET_TR, reverse=True)
    with open(str(output_path), mode="w") as out_file:
        with open(str(input_path), mode="r") as in_file:
            for line in in_file:
                line = line.strip().split()

                syn = wn.lemma_from_key(line[1]).synset()
                syn = "wn:" + str(syn.offset()).zfill(8) + syn.pos()
                syn = _get_frequent_sense(syn, wn2bn, bn2domain, is_bn)

                out_file.write(line[0] + " " + syn + "\n")


def _get_frequent_sense(
    syn: str, wn2bn: Dict[str, str], bn2domain: Dict[str, str], is_bn: bool
) -> str:
    """
    This method is used to compute the most frequent sense
    :param syn: synset to compute
    :param wn2bn: a map Wordnet to Babelnet
    :param bn2domain: a map Babelnet to domain
    :param is_bn: if True, it returns a Babelnet synset
                if False, it returns in coarse-grained synset
    :return: the most frequent sense
    """
    syn = wn2bn.get(syn) if is_bn else bn2domain.get(wn2bn.get(syn))
    if syn is None:
        domain2bn = reverse_vocab(bn2domain)
        syn, _ = FreqDist(list(domain2bn.keys())).most_common(1)[0]
    return syn


def domains_converters(corpus: List[List[str]], bn2domain: Dict[str, str]) -> List[str]:
    """
    This method is used to convert a corpus from Babelnet format to a domain
    :param corpus: corpus to convert
    :param bn2domain: a map Babelnet to domain
    :return: converted corpus
    """
    for sentences in corpus:
        yield _domain_converter(sentences, bn2domain)


def _domain_converter(sentences: List[str], bn2domain: Dict[str, str]) -> List[str]:
    """
    This method is used to convert sentences from Babelnet formato to a domain format.
    Moreover, it prints how many missed words there were
    :param sentences: sentences to convert
    :param bn2domain: a map Babelnet to domain
    :return: converted sentences
    """
    out = []
    i, j = 0, 0
    domain2bn = reverse_vocab(bn2domain)
    mfs, _ = FreqDist(list(domain2bn.keys())).most_common(1)[0]

    for sentence in sentences:
        tmp = []
        for word in sentence.split():

            if "_bn:" in word:
                word1 = word.split("_")
                word = bn2domain.get(word1[-1])
                if word is None:
                    word = "<UNK>"
                    i += 1
                j += 1

            tmp.append(word)
        out.append(" ".join(tmp))

    print("\t\tUnk", i, "Tot", j, "rate", str(round(i / j * 100, 2)) + "%")
    return out


def convert_pos(param):
    """
    This method is used to convert the Raganato'f format pos to Wordnet format pos
    :param param: pos to convert
    :return: converted pos
    """
    converter = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}

    return converter.get(param, "n")


def convert_sens_emb(sentences, labels):
    """
    This method is used to convert sentences in a Babelnet format to a
    sense embeddings sentence format
    :param sentences: sentences to convert
    :param labels: label where retrieve senses
    :return: converted sentences
    """
    bn2wn = read_map(config.BABELNET2WORDNET_TR)
    return [" ".join(input_sens_emb([labels[i]], bn2wn)) for i in range(len(sentences))]


def squeeze_emb(emb: gensim.models.word2vec, restricted: Set):
    """
    This method is used to retrieve from w2v pretrained embeddings
    only the necessary words
    :param emb: w2v embeddings
    :param restricted: sentences
    :return: a squeezed embedding
    """
    vectors, new_index2entity, norm_vect = [], [], []
    new_vocab = {}

    for i in range(len(emb.vocab)):
        word = emb.index2entity[i]
        vec = emb.vectors[i]
        vocab = emb.vocab[word]
        vec_norm = emb.vectors_norm[i] if emb.vectors_norm else []
        if word in restricted:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            vectors.append(vec)
            if vec_norm:
                norm_vect.append(vec_norm)

    emb.vocab = new_vocab
    emb.vectors = np.array(vectors)
    emb.index2entity = np.array(new_index2entity)
    emb.index2word = np.array(new_index2entity)
    if norm_vect:
        emb.vectors_norm = np.array(norm_vect)
    return emb


def emb_txt_to_bin(path_input: str, path_output: str) -> None:
    """
    This method is used to convert a w2v pretrained embedding from
    txt form to bin form. Is used to reduce it dimension
    :param path_input: input path
    :param path_output: output path
    :return: None
    """
    emb = gensim.models.KeyedVectors.load_word2vec_format(path_input, binary=False)
    emb.save_word2vec_format(path_output, binary=True)


def check_integrity(sentences: List[str], labels: List[str]) -> bool:
    """
    This method is used to check whether the sentences have the same length
    :param sentences: training sentences
    :param labels: labels sentences
    :return: True if so,
            False otherwise
    """
    for i in range(len(sentences)):
        if len(sentences[i].split()) != len(labels[i].split()):
            return False
    return True
