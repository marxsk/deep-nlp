""" NLP pipeline for experiments in Czech language

    @param sys.argv[1] - Name of the file where first line is read and parsed
"""
import logging
import os
import sys

from majka import Majka
from nltk import sent_tokenize, word_tokenize

MAJKA_WLT_PATH = "majka/majka.w-lt"
LOGLEVEL_DEFAULT = "DEBUG"

BLOCKED_LEMMA = ["dobřit"]

MORPH = Majka(MAJKA_WLT_PATH)
LOGGER = logging.getLogger('deep-nlp-pipeline')


def local_morph(word):
    """ Return morphology info for words unknown to the default analyzer

        @param word - Word to recognize
        @return list of morphology analyses or empty list if word is unknown
    """
    known_words = {
        '.': [{'lemma': '.', 'tags': {'pos': 'interpunction'}}],
        ',': [{'lemma': ',', 'tags': {'pos': 'interpunction'}}]
    }
    return known_words[word] if word in known_words else []


def local_blocklist(analyses):
    """ Remove analyses that we believe are wrong because those lemmas are never used """
    result = []
    for a in analyses:
        if not a['lemma'] in BLOCKED_LEMMA:
            result.append(a)
    return result


def parse_document(text):
    """ Parse document and show results on standard output

        @param text - Document (several sentences) to parse
    """
    # get sentences
    for sentence in sent_tokenize(text, language='czech'):
        for word in word_tokenize(sentence):
            res = MORPH.find(word)
            if res == []:
                res = local_morph(word)
                if res == []:
                    LOGGER.debug('Unknown token detected "%s"', word)

            if res:
                for candidate in res:
                    if candidate['tags'] == {}:
                        LOGGER.debug(
                            'Token "%s" was recognized but it has no tags at all', word)
            res = local_blocklist(res)
            print(res)
        print("---")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", LOGLEVEL_DEFAULT))

    with open(sys.argv[1], 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            parse_document(line)
