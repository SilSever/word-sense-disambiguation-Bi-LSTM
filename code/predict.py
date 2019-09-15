import os
from typing import Dict

import gensim
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import config
import models
import parser
import preprocesser
import utils

"""
    :author Silvio Severino
"""


def predict_babelnet(input_path: str, output_path: str, resources_path: str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    print("Predicting Babelnet...")
    out_vocab = utils.read_vocabulary(os.path.join(resources_path, config.OUT_VOCAB_BN))
    bn2wn = utils.read_map(
        os.path.join(resources_path, config.BABELNET2WORDNET), reverse=False
    )

    dense_layer = 0
    is_bn = True

    _prediction(
        input_path,
        output_path,
        resources_path,
        out_vocab,
        dense_layer,
        is_bn,
        bn2domain=bn2wn,
    )


def predict_wordnet_domains(
    input_path: str, output_path: str, resources_path: str
) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    print("Predicting Wordnet Domains...")
    out_vocab = utils.read_vocabulary(
        os.path.join(resources_path, config.OUT_VOCAB_WND)
    )
    bn2wnd = utils.read_map(
        os.path.join(resources_path, config.BABELNET2WNDOMAINS), reverse=False
    )
    dense_layer = 1
    is_bn = False

    _prediction(
        input_path,
        output_path,
        resources_path,
        out_vocab,
        dense_layer,
        is_bn,
        bn2domain=bn2wnd,
    )


def predict_lexicographer(
    input_path: str, output_path: str, resources_path: str
) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    print("Predicting Lexicographer...")
    out_vocab = utils.read_vocabulary(
        os.path.join(resources_path, config.OUT_VOCAB_LEX)
    )
    bn2lex = utils.read_map(
        os.path.join(resources_path, config.BABELNET2LEXANAMES), reverse=False
    )
    dense_layer = 2
    is_bn = False

    _prediction(
        input_path,
        output_path,
        resources_path,
        out_vocab,
        dense_layer,
        is_bn,
        bn2domain=bn2lex,
    )


def _prediction(
    input_path: str,
    output_path: str,
    resources_path: str,
    out_vocab: Dict,
    i_dense: int,
    is_bn: bool,
    bn2domain: Dict = None,
) -> None:
    """
    This method is used to handle the prediction of a task
    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :param out_vocab: sense inventory
    :param i_dense: i-esime fully connected layer:
                    0 -> Babelnet
                    1 -> Wordnet domains
                    2 -> Lexicographer
    :param is_bn: if True, predicts Babelnet
    :param bn2domain: a map Babelnet to domain
    :return: None
    """
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config_tf))

    test_set = parser.parser_test_set(input_path)

    vocab = utils.read_vocabulary(os.path.join(resources_path, config.VOCAB))
    wn2bn = utils.read_map(
        os.path.join(resources_path, config.BABELNET2WORDNET), reverse=True
    )

    out_vocab_bn = utils.read_vocabulary(
        os.path.join(resources_path, config.OUT_VOCAB_BN)
    )
    out_vocab_wnd = utils.read_vocabulary(
        os.path.join(resources_path, config.OUT_VOCAB_WND)
    )
    out_vocab_lex = utils.read_vocabulary(
        os.path.join(resources_path, config.OUT_VOCAB_LEX)
    )
    out_vocab_pos = utils.read_vocabulary(
        os.path.join(resources_path, config.POS_VOCAB)
    )

    pre_trained = gensim.models.KeyedVectors.load_word2vec_format(
        os.path.join(resources_path, config.SQUEEZED_EMB), binary=True
    )

    print("Downloading ELMo...")
    model = models.build_model(
        vocab_size=len(vocab),
        out_size_bn=len(out_vocab_bn),
        out_size_wnd=len(out_vocab_wnd),
        out_size_lex=len(out_vocab_lex),
        out_size_pos=len(out_vocab_pos),
        word2vec=pre_trained,
        is_elmo=config.IS_ELMO,
        attention=config.ATTENTION,
        is_sense_emb=config.SENSE_EMB,
    )

    reversed_vocab = utils.reverse_vocab(out_vocab_bn)
    model.load_weights(str(os.path.join(resources_path, config.MODEL_WEIGHTS)))

    with open(str(output_path), mode="w") as file:
        for row in tqdm(test_set):

            if not config.IS_ELMO:
                tmp = preprocesser.text2id([row[0]], vocab)
            else:
                tmp = np.array([row[0]])

            tmp_row = list(row[2])

            inp = tmp
            if config.SENSE_EMB:
                sens_emb = np.ones((1, len(inp[0].split())), dtype=int)
                inp_pos = np.array([row[1]])
                inp = [inp, sens_emb, inp_pos]

            predictions = model.predict(inp, verbose=0)[i_dense]

            for senses in tmp_row:

                sense_position = utils.senses_position_from_vocab(
                    senses["lemma"], out_vocab_bn, bn2domain
                )

                synsets = [reversed_vocab[x] for x in sense_position]

                if not is_bn:
                    synsets = [bn2domain.get(syn.split("_")[-1]) for syn in synsets]
                    sense_position = [out_vocab[syn] for syn in synsets]

                to_compute = np.array(
                    [
                        predictions[0][senses["position"]][sen_pos]
                        for sen_pos in sense_position
                    ]
                )

                if len(to_compute) != 0:
                    file.write(
                        senses["id"]
                        + " "
                        + synsets[to_compute.argmax()].split("_")[-1]
                        + "\n"
                    )
                else:
                    file.write(
                        senses["id"]
                        + " "
                        + utils.most_frequent_sense(
                            senses["lemma"],
                            senses["pos"],
                            wn2bn,
                            bn2domain=bn2domain,
                            is_bn=is_bn,
                        )
                        + "\n"
                    )


def predicter() -> None:
    """
    This method is used to predict all the test sets
    :return: None
    """
    if not os.path.exists(config.PREDICT_FOLDER):
        os.mkdir(config.PREDICT_FOLDER)

    for subdir, dirs, files in os.walk(str(config.TEST_SETS)):
        for file in files:
            if file.endswith("data.xml"):
                input_path = os.path.join(subdir, file)
                file_name = file.split(".")[0]
                out_path = (
                    str(config.PREDICT_BABELNET)
                    + "_"
                    + file_name
                    + str(config.PREDICT_EXTENSION)
                )
                predict_babelnet(input_path, out_path, config.RESOURCE_DIR)

                out_path = (
                    str(config.PREDICT_WORDNET_DOMAINS)
                    + "_"
                    + file_name
                    + str(config.PREDICT_EXTENSION)
                )
                predict_wordnet_domains(input_path, out_path, config.RESOURCE_DIR)

                out_path = (
                    str(config.PREDICT_LEXICOGRAPHER)
                    + "_"
                    + file_name
                    + str(config.PREDICT_EXTENSION)
                )
                predict_lexicographer(input_path, out_path, config.RESOURCE_DIR)


def scorer() -> None:
    """
    This method is used to compute all the F1 scores, using Raganato's evaluation framework
    :return: None
    """
    write_score = " >> " + str(config.PREDICT_SCORE)
    with open(str(config.PREDICT_SCORE), mode="w"):
        for subdir, dirs, files in os.walk(str(config.TEST_SETS)):
            for file in files:

                folder = file.split("_")[0]
                gold_key = os.path.join(subdir, file)
                domain = ""
                if "_babelnet" in file:
                    domain = "babelnet_"
                if "_lexico" in file:
                    domain = "lexico_"
                if "_wndom" in file:
                    domain = "wndom_"

                if domain != "":
                    to_predict = os.path.join(
                        str(config.PREDICT_FOLDER),
                        domain + folder + str(config.PREDICT_EXTENSION),
                    )
                    os.system("echo " + file + write_score)
                    os.system(
                        "java -cp "
                        + str(config.TEST_SETS)
                        + " Scorer "
                        + gold_key
                        + " "
                        + to_predict
                        + write_score
                    )
                    os.system("echo " + write_score)
