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

MAJKA_WLT_PATH = "majka/majka.w-lt"
LOGLEVEL_DEFAULT = "INFO"

BLOCKED_LEMMA = ["dobřit"]
BLOCKED_K1 = ["malá"]

MORPH = Majka(MAJKA_WLT_PATH)
LOGGER = logging.getLogger('deep-nlp-pipeline')

TYPES_MEASURE = ["pravděpodobně", "velice", "vždy", "dobře", "top", "naprostý", "perfektní",
                 "hezky", "spousta", "super", "výborný", "vymakaný", "samý",
                 "velmi", "všechno", "lehce", "fakt", "ideální", "vše", "maximálně", "dobrý",
                 "nadměrný", "nejlepší", "skvělý", "krásně", "pořádný",
                 "velký", "výrazný", "dokonale", "celkem", "ok", "podstatný", "kladně",
                 "neustále", "nový", "ohromně", "veškerý"]

TYPES_QUALITY = ["přehledný", "snadný", "intuitivní", "jasný", "rychlý", "funkční", "ovladatelný",
                 "jednoduchý", "příjemný", "pěkný", "moderní", "povedený", "spolehlivý", "logický",
                 "pohodlný", "chytrý", "komfortní", "šikovný", "pochopitelný", "zdařilý", "čistý",
                 "svižný", "zabezpečený", "praktický", "užitečný", "optimalizovaný"]

TYPES_APP = ["aplikace", "web", "tlačítko", "bankovnictví", "banka", "app", "apka", "transakce",
             "touchID", "faceID", "graf", "platba", "autorizace"]

TYPES_ATTR_APP = ["reklama", "problém", "spolehlivost", "funkce", "stabilita", "pozitivum", "použití",
                  "používání", "grafika", "design", "přehlednost", "jednoduchost", "aktualizace",
                  "prostředí", "ovládání", "vylepšení", "vzhled", "využití", "funkčnost", "rychlost",
                  "věc", "zapínání", "vypínání", "update", "ovládání"]

RE_EMOTICONS = re.compile(u'['
                          u'\U0001F300-\U0001F64F'
                          u'\U0001F680-\U0001F6FF'
                          u'\u2600-\u26FF\u2700-\u27BF]+',
                          re.UNICODE)

ALLOWED_TERMINALS = ["a", ","]

GRAMMAR = """
    // eps_* -> TERMINAL alebo epsilon 
    // *_single -> NETERMINAL bez koordinacii

    sentence: t_app | t_attr

    t_app: (t_quality eps_app) | (APP t_quality)

    t_attr: (t_quality ATTR eps_app) | (ATTR eps_app t_quality)

    t_quality: t_quality_single | ((t_quality_single ",")+ t_quality_single) | ((t_quality_single ",")* t_quality_single "a" t_quality_single)
    t_quality_single: (QUALITY) | (t_measure eps_quality)

    t_measure: MEASURE | MEASURE MEASURE

    eps_quality: QUALITY | empty
    eps_app: APP | empty

    empty:

    MEASURE: "#measure"
    APP: "#app"
    QUALITY: "#quality"
    ATTR: "#attr"

    %ignore " "
"""
PARSER = Lark(GRAMMAR, parser='earley', start='sentence',
              debug=True, ambiguity='explicit')

sentence_counter = 0


def run_earley_parser(sentence, counter, variant, label, directory):
    if '#unknown' in sentence:
        # Unknown token cannot be resolved into valid tree
        return None

    try:
        parse_tree = PARSER.parse(" ".join(sentence))
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


def add_semtypes_for_lemma(lemma):
    """ Return string representation of list of all semantic types for given lemma """
    # @todo: is POS required for correct semtype(?)
    all_possible_types = []
    if lemma in TYPES_MEASURE:
        all_possible_types.append("#measure")
    if lemma in TYPES_QUALITY:
        all_possible_types.append("#quality")
    if lemma in TYPES_APP:
        all_possible_types.append("#app")
    if lemma in TYPES_ATTR_APP:
        all_possible_types.append("#attr")
    return ":".join(all_possible_types)


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
        '!': [{'lemma': '!', 'tags': {'pos': 'interpunction'}}],
        ',': [{'lemma': ',', 'tags': {'pos': 'interpunction'}}],
        '(': [{'lemma': '(', 'tags': {'pos': 'parentheses'}}],
        ')': [{'lemma': ')', 'tags': {'pos': 'parentheses'}}],
        "OK": [{'lemma': 'ok', 'tags': {'pos': 'abbreviation'}}]
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

    # get sentences
    for sentence in sent_tokenize(text, language='czech'):
        contain_verb = False
        valid_sentence = True
        tokens = []

        sentence_without_emoticons = RE_EMOTICONS.sub('', sentence)
        sentence_counter += 1

        for word in word_tokenize(sentence_without_emoticons):
            res = MORPH.find(word) + local_morph(word)

            if res == []:
                valid_sentence = False
                LOGGER.debug('Unknown token detected "%s"', word)

            if res:
                for candidate in res:
                    if candidate['tags'] == {}:
                        valid_sentence = False
                        LOGGER.debug(
                            'Token "%s" was recognized but it has no tags at all', word)
            res = local_blocklist(res)
            for analyse in res:
                analyse['semtype'] = add_semtypes_for_lemma(analyse['lemma'])
                # @note: ignore all sentences with verb
                if analyse.get('tags', {}).get('pos', '') == 'verb':
                    contain_verb = True

            tokens.append(res)
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
            if new_sentence[-1] in [["."], ["!"]]:
                new_sentence.pop()

            cfg_sentence = []
            for token_analysis in new_sentence:
                cfg_sentence.append(list(set([normalize_sem_token(token)
                                              for token in token_analysis])))

            print(">>>")
            print(cfg_sentence)

            if ["#unknown"] in cfg_sentence:
                # sentences that cannot be desambiguated because at least one word is completely unknown
                LOGGER.error(new_sentence)
                continue

#            print(sentence)
#            print(new_sentence)
#            print(cfg_sentence)

            # create all combinations that we have to parse
            variant = 1
            for c in itertools.product(*cfg_sentence):
                if c:
                    success = run_earley_parser(c, sentence_counter, variant,
                                                sentence_without_emoticons, output_directory)
                    if success:
                        variant += 1

#            print("----sentence----")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", LOGLEVEL_DEFAULT))

    with open(sys.argv[1], 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            parse_document(line, sys.argv[2])
