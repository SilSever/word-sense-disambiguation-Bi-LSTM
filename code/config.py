import pathlib

# directories
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
RESOURCE_DIR = pathlib.Path(__file__).resolve().parent.parent / "resources"
# DATA_DIR = pathlib.Path("data")
# RESOURCE_DIR = pathlib.Path("resources")
SENTENCES = DATA_DIR / "Sentences"
EVALUATION_FRAMEWORK = DATA_DIR / "WSD_Evaluation_Framework"
TEST_SETS = EVALUATION_FRAMEWORK / "Evaluation_Datasets"
TRAINING_SETS = EVALUATION_FRAMEWORK / "Training_Corpora"
PLOT_FOLDER = RESOURCE_DIR / "plots"
PREDICT_FOLDER = RESOURCE_DIR / "predictions"
DRIVE_DIR = pathlib.Path("drive/My Drive/nlp")

# model
MODEL = RESOURCE_DIR / "wsd.model"
MODEL_TRY = RESOURCE_DIR / "wsd_try.model"
SENSE_EMBEDDINGS = RESOURCE_DIR / "embeddings.vec"
SQUEEZED_EMB = "squeezed_embeddings.vec"
EUROTOM_EMBEDDINGS = RESOURCE_DIR / "embeddings_eurotom.vec"
MODEL_CHECK_POINT = RESOURCE_DIR / "model_checkpoint.h5"
RICHARD_MODEL = RESOURCE_DIR / "model_adagrad.h5"
MODEL_WEIGHTS = "model_weights.h5"
MODEL_WEIGHTS_TR = RESOURCE_DIR / "model_weights.h5"
SQUEEZED_EMB_TR = RESOURCE_DIR / "squeezed_embeddings.vec"
IS_ELMO = True
ATTENTION = True
SENSE_EMB = False

# mapping file
BABELNET2LEXANAMES = "babelnet2lexnames.tsv"
BABELNET2WNDOMAINS = "babelnet2wndomains.tsv"
BABELNET2WORDNET = "babelnet2wordnet.tsv"
BABELNET2LEXANAMES_TR = RESOURCE_DIR / "babelnet2lexnames.tsv"
BABELNET2WNDOMAINS_TR = RESOURCE_DIR / "babelnet2wndomains.tsv"
BABELNET2WORDNET_TR = RESOURCE_DIR / "babelnet2wordnet.tsv"


# vocabularies
VOCAB = "vocab.txt"
SENSE_VOCAB = RESOURCE_DIR / "sense_emb.txt"
POS_VOCAB = "pos_emb.txt"
OUT_VOCAB_BN = "out_vocab_bn.txt"
OUT_VOCAB_WND = "out_vocab_wnd.txt"
OUT_VOCAB_LEX = "out_vocab_lex.txt"
VOCAB_TR = RESOURCE_DIR / "vocab.txt"
POS_VOCAB_TR = RESOURCE_DIR / "pos_emb.txt"
OUT_VOCAB_BN_TR = RESOURCE_DIR / "out_vocab_bn.txt"
OUT_VOCAB_WND_TR = RESOURCE_DIR / "out_vocab_wnd.txt"
OUT_VOCAB_LEX_TR = RESOURCE_DIR / "out_vocab_lex.txt"


# parsed sentences
SEMCOR_SENTENCES = SENTENCES / "semcor_sentences.txt"
ALL_SENTENCES = SENTENCES / "ALL_sentences.txt"
SEMEVAL2007_SENTENCES = SENTENCES / "semeval2007_sentences.txt"
SEMEVAL2013_SENTENCES = SENTENCES / "semeval2013_sentences.txt"
SEMEVAL2015_SENTENCES = SENTENCES / "semeval2015_sentences.txt"
SENSEVAL2_SENTENCES = SENTENCES / "senseval2_sentences.txt"
SENSEVAL3_SENTENCES = SENTENCES / "senseval3_sentences.txt"


# predicted sentences
PREDICT_BABELNET = PREDICT_FOLDER / "babelnet"
PREDICT_WORDNET_DOMAINS = PREDICT_FOLDER / "wndom"
PREDICT_LEXICOGRAPHER = PREDICT_FOLDER / "lexico"
PREDICT_EXTENSION = ".key"
PREDICT_SCORE = PREDICT_FOLDER / "scores.txt"
RICHARD_BABELNET = RESOURCE_DIR / "richard_babelnet.key"
