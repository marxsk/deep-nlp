""" NLP pipeline for experiments in Czech language

    @param sys.argv[1] - Name of the file where first line is read and parsed
"""
import itertools
import logging
import os
import re
import sys

from lark import Lark, tree
from majka import Majka
from nltk import sent_tokenize, word_tokenize

from preprocessor import preprocessor

MAJKA_WLT_PATH = "majka/majka.w-lt"
VOCABULARY_PATH = "vocabulary.csv"
LOGLEVEL_DEFAULT = "DEBUG"

BLOCKED_LEMMA = ["dobřit"]
BLOCKED_K1 = ["malá"]

MORPH = Majka(MAJKA_WLT_PATH)
LOGGER = logging.getLogger('deep-nlp-pipeline')

RE_EMOTICONS = re.compile(u'['
                          u'\U0001F300-\U0001F64F'
                          u'\U0001F680-\U0001F6FF'
                          u'\u2600-\u26FF\u2700-\u27BF]+',
                          re.UNICODE)

ALLOWED_TERMINALS = ["a", ",", "s"]

GRAMMAR = """
    // eps_* -> TERMINAL alebo epsilon; je generovana automaticky pre kazdy terminal
    // *_single -> NETERMINAL bez koordinacii ; koordinacia je vygenerovana automaticky
    // *_req -> vyžaduje naplnenie argumentu, aby sa dalo použiť vo vete

    sentence: t_app
    sentence: t_attr_complex
    sentence: valency_foo_val

    t_app: (t_quality eps_app) | (APP t_quality)

    t_attr_complex: (t_attr eps_app) | (t_attr eps_app t_quality)
    t_attr_single: t_quality* ATTR

    t_quality_single: (QUALITY) | (t_measure eps_quality) | (t_measure_req QUALITY)

    t_measure_req: (MEASURE_REQ)
    t_measure: (MEASURE) | (MEASURE MEASURE) | (MEASURE_REQ MEASURE)

    valency_foo_val: (t_foo_val empty_prep_s eps_app) | (t_foo_val prep_s APP)
    t_foo_val: t_measure? FOO_VAL_S

    prep_s: "s"
    empty_prep_s: empty

"""
sentence_counter = 0


def run_earley_parser(sentence, counter, variant, label, directory):
    if '#unknown' in sentence:
        # Unknown token cannot be resolved into valid tree
        return None

    try:
        sentence_wo_floskule = [x for x in sentence if x != "#floskule"]
        parse_tree = PARSER.parse(" ".join(sentence_wo_floskule))
#        print(sentence)
#        print(parse_tree.pretty())

        tree.pydot__tree_to_png(
            parse_tree, directory + '/sentence-{:03d}-{:02d}.png'.format(counter, variant), label=label + "\n" + " ".join(sentence))
        with open(directory + "/sentence-{:03d}-{:02d}.pretty".format(counter, variant), "w") as f:
            f.write(parse_tree.pretty())

    except Exception as e:
        LOGGER.info(e)
        LOGGER.info("Unable to create a tree for <%s>", (" ".join(sentence)))
        return False

    return True


def load_vocabulary():
    """ Load vocabulary of TERMINAL:WORD items """
    vocabulary = {}

    with open(VOCABULARY_PATH, "r") as f:
        for line in f.readlines():
            (semtype, word) = line.strip().split(":")
            if not word in vocabulary:
                vocabulary[word] = set()
            vocabulary[word].add(semtype)

    return vocabulary


def load_semtypes_from_vocabulary():
    """ Load all semantic types from the vocabulary """
    vocabulary = load_vocabulary()
    semcabulary = dict()

    for word in vocabulary:
        semtypes = list(vocabulary[word])
        semtypes.sort()
        semcabulary["^".join(semtypes)] = 1

    return semcabulary


def add_semtypes_for_lemma(vocabulary, lemma):
    """ Return string representation of list of all semantic types for given lemma """
    # @todo: is POS required for correct semtype(?)
    all_possible_types = []
    for semtype in vocabulary.get(lemma, []):
        all_possible_types.append(semtype)
    return all_possible_types


def normalize_sem_token(token):
    """ Create a normalized token/semtypes used for CFG """
    if token.startswith('#'):
        return token
    elif token in ALLOWED_TERMINALS:
        return token
    else:
        return "#unknown_" + token


def local_morph(word):
    """ Return morphology info for words unknown to the default analyzer

        @param word - Word to recognize
        @return list of morphology analyses or empty list if word is unknown
    """
    known_words = {
        '.': [{'lemma': '.', 'tags': {'pos': 'interpunction'}}],
        '...': [{'lemma': '...', 'tags': {'pos': 'interpunction'}}],
        '!': [{'lemma': '!', 'tags': {'pos': 'interpunction'}}],
        ',': [{'lemma': ',', 'tags': {'pos': 'interpunction'}}],
        '(': [{'lemma': '(', 'tags': {'pos': 'parentheses'}}],
        ')': [{'lemma': ')', 'tags': {'pos': 'parentheses'}}],
        "ok": [{'lemma': 'ok', 'tags': {'pos': 'abbreviation'}}]
    }
    return known_words[word] if word in known_words else []


def local_blocklist(analyses):
    """ Remove analyses that we believe are wrong because those lemmas are never used """
    result = []
    for analyse in analyses:
        if analyse['lemma'] in BLOCKED_LEMMA:
            continue
        if analyse['lemma'] in BLOCKED_K1 and analyse.get('tags', {}).get('pos', '') == 'substantive':
            continue
        result.append(analyse)
    return result


def parse_document(text, output_directory):
    """ Parse document and show results on standard output

        @param text - Document (several sentences) to parse
    """
    global sentence_counter

    vocabulary = load_vocabulary()

    # get sentences
    for sentence in sent_tokenize(text, language='czech'):
        LOGGER.info("**** Begin of the sentence parsing")

        contain_verb = False
        valid_sentence = True
        tokens = []
        success_combinations = []

        sentence_without_emoticons = RE_EMOTICONS.sub('', sentence)
        sentence_counter += 1

        for word in word_tokenize(sentence_without_emoticons):
            res = MORPH.find(word) + local_morph(word)

            if res == []:
                valid_sentence = False
                LOGGER.debug('Unknown token detected "%s"', word)

            if res:
                for candidate in res:
                    if candidate['tags'] == {} and candidate['lemma'] != 's':
                        valid_sentence = False
                        LOGGER.debug(
                            'Token "%s" was recognized but it has no tags at all', word)
            res = local_blocklist(res)
            for analyse in res:
                analyse['semtype'] = add_semtypes_for_lemma(
                    vocabulary, analyse['lemma'])
                # @note: ignore all sentences with verb
                if analyse.get('tags', {}).get('pos', '') == 'verb':
                    contain_verb = True

            # unpack semtypes from string to multiple elements
            unpack_res = []
            for analyse in res:
                if analyse['semtype']:
                    for semtype in analyse['semtype']:
                        new_analyses = dict(analyse)
                        new_analyses['semtype'] = semtype
                        unpack_res.append(new_analyses)
                else:
                    new_analyses = dict(analyse)
                    unpack_res.append(new_analyses)

            tokens.append(unpack_res)

        if not contain_verb and valid_sentence and tokens:
            new_sentence = []
            for token in tokens:
                token_analysis = []
                for analysis in token:
                    base_form = analysis['semtype'] if analysis['semtype'] else analysis.get(
                        'lemma')
                    token_analysis.append(base_form)
                new_sentence.append(list(set(token_analysis)))

            # remove trailing punctuation
            if new_sentence[-1] in [["."], ["!"], ["..."]]:
                new_sentence.pop()

            cfg_sentence = []
            for token_analysis in new_sentence:
                cfg_sentence.append(list(set([normalize_sem_token(token)
                                              for token in token_analysis])))

            if ["#unknown"] in cfg_sentence:
                # sentences that cannot be desambiguated because at least one word is completely unknown
                LOGGER.error(new_sentence)
                continue

            # create all combinations that we have to parse
            variant = 1
            print(cfg_sentence)
            for c in itertools.product(*cfg_sentence):
                if c:
                    success = run_earley_parser(c, sentence_counter, variant,
                                                sentence_without_emoticons, output_directory)
                    if success:
                        variant += 1
                        success_combinations.append(c)

        if not success_combinations:
            print(">>>")
            print(sentence)
            print(success_combinations)

        LOGGER.info("**** End of the sentence parsing\n\n\n\n\n")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", LOGLEVEL_DEFAULT))

    PARSER = Lark(preprocessor(GRAMMAR, load_semtypes_from_vocabulary()), parser='earley', start='sentence',
                  debug=True, ambiguity='explicit')

    with open(sys.argv[1], 'r', encoding='utf-8') as fh:
        for input_line in fh.readlines():
            parse_document(input_line, sys.argv[2])
