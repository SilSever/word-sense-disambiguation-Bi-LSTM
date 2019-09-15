import os
from typing import List, Tuple, Dict

from lxml.etree import iterparse
from nltk.corpus import wordnet as wn
from tqdm import tqdm

import config
import utils

"""
    :author Silvio Severino
"""


def goldkey2bn(
    sense_id: str, key2sensekey: Dict[str, str], wn2bn: Dict[str, str]
) -> str:
    """
    A simple method to convert the gold key id to a Babelnet synset
    :param sense_id: sense id
    :param key2sensekey: gold key map
    :param wn2bn: map Wordnet to Babelnet
    :return: the Babelnet synset
    """
    synset = wn.lemma_from_key(key2sensekey[sense_id]).synset()
    wn_syn = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
    return wn2bn.get(wn_syn)


def parser_raganato_format(
    filepath: str, key2sensekey: Dict[str, str], wn2bn: Dict[str, str]
) -> List[Tuple[str, str, str]]:
    """
    This method is used to parse an XML having Raganato's format
    :param filepath: path from read
    :param key2sensekey: gold key map
    :param wn2bn: map Wordnet to Babelnet
    :return:
        the non-disambiguated sentences
        the disambiguated sentences
        the pos sentences
    """
    out_sentences = []
    tmp_train_x, tmp_train_y, tmp_train_pos = "", "", ""

    context = iterparse(str(filepath), tag=("sentence", "wf", "instance"))
    for event, elem in tqdm(context, desc="Parsing " + str(filepath).split("/")[-1]):

        if elem.tag == "wf":
            if not elem.attrib["pos"] == ".":
                tmp_train_x += "_".join(elem.text.lower().split()) + " "
                tmp_train_y += elem.text.lower() + " "
                tmp_train_pos += elem.attrib["pos"].lower() + " "

        elif elem.tag == "instance":
            if not elem.attrib["pos"] == ".":
                tmp_train_x += "_".join(elem.text.lower().split()) + " "
                tmp_train_y += (
                    elem.attrib["lemma"].lower()
                    + "_"
                    + goldkey2bn(elem.attrib["id"], key2sensekey, wn2bn)
                    + " "
                )
                tmp_train_pos += elem.attrib["pos"].lower() + " "

        elif event == "end" and elem.tag == "sentence":
            if tmp_train_x != "" and tmp_train_y != "" and tmp_train_pos != "":
                out_sentences.append(
                    (tmp_train_x[:-1], tmp_train_y[:-1], tmp_train_pos[:-1])
                )
            tmp_train_x, tmp_train_y, tmp_train_pos = "", "", ""

        # fast iter, clear the memory
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    return out_sentences


def _parse_all_corpus(corpus_path: str, wn2bn: Dict[str, str]) -> None:
    """
    A method to parse all available corpus
    :param corpus_path: corpus folder path
    :param wn2bn: map Wordnet to Babelnet
    :return: None
    """
    for subdir, dirs, files in os.walk(str(corpus_path)):

        data_path, gold_path, parsed_path = "", "", ""

        for file in files:
            if file.endswith("data.xml"):
                data_path = os.path.join(subdir, file)
            elif file.endswith("gold.key.txt"):
                gold_path = os.path.join(subdir, file)

            # if the corpus is not parsed yet
            parsed_path = os.path.join(
                config.SENTENCES, file.split(".")[0] + "_sentences.txt"
            )
            if not os.path.isfile(parsed_path) and all(
                (path != "") for path in [data_path, gold_path]
            ):
                key_map = utils.read_map(gold_path, delimiter=" ")
                utils.write_sentences_and_labels(
                    parsed_path, parser_raganato_format(data_path, key_map, wn2bn)
                )


def parser_test_set(filepath: str) -> List[Tuple[str, str, List[dict]]]:
    """
    This method is used to parse an XML having Raganato's format
    :param filepath: path of test set
    :return:
        a List of Tuples having:
            - the parsed non-disambiguated sentence
            - the parsed pos sentence
            - A List of Map having all the senses information: id, lemma, pos, position
    """
    out_sentences, pos_id = [], []
    tmp_test_x, tmp_pos = "", ""
    pos_counter, j = 0, 0
    context = iterparse(str(filepath), tag=("sentence", "wf", "instance"))
    for event, elem in tqdm(context, desc="Parsing " + str(filepath).split("/")[-1]):

        if elem.tag == "wf":
            if not elem.attrib["pos"] == ".":
                tmp_test_x += "_".join(elem.text.lower().split()) + " "
                tmp_pos += elem.attrib["pos"].lower() + " "
            else:
                j += 1

        elif elem.tag == "instance":
            if not elem.attrib["pos"] == ".":
                tmp_test_x += "_".join(elem.text.lower().split()) + " "
                tmp_pos += elem.attrib["pos"].lower() + " "
                pos_id.append(
                    {
                        "id": elem.attrib["id"],
                        "lemma": elem.attrib["lemma"].lower(),
                        "pos": utils.convert_pos(elem.attrib["pos"]),
                        "position": pos_counter,
                    }
                )
            else:
                j += 1

        elif event == "end" and elem.tag == "sentence":
            if tmp_test_x != "" and tmp_pos != "":
                out_sentences.append((tmp_test_x[:-1], tmp_pos[:-1], pos_id))
            pos_id = []
            tmp_test_x, tmp_pos = "", ""
            pos_counter = -1
        pos_counter += 1 - j
        j = 0

        # fast iter, clear the memory
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    return out_sentences


def parse_all():
    """
    A simple method used to handle both the corpus and the test sets
    :return: None
    """
    wn2bn = utils.read_map(config.BABELNET2WORDNET_TR, reverse=True)

    _parse_all_corpus(config.TEST_SETS, wn2bn)
    _parse_all_corpus(config.TRAINING_SETS, wn2bn)
