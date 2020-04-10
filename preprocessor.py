"""
Preprocessor transforming "our grammar" to the lark format

Be aware that "terminal" is the node that is matched directly to the Token(),
it has to be written in the upppercase.

Also 'semantic type' / 'semantic class' and 'token' are synonyms in this context.
"""

import logging

GENERATED_LINE = -1
LOGGER = logging.getLogger('deep-nlp-pipeline:preprocessor')


def _add_epsilon_for_each_terminal(rules_by_line):
    """ Add epsilon and empty non-terminal for each of the terminals: eps_T -> T | empty """
    epsilon_terminals = []

    for left_side in rules_by_line:
        if left_side.upper() == left_side:
            eps_terminal_name = left_side.lower()
            epsilon_terminals.append('eps_%s: %s | empty_%s' %
                                     (eps_terminal_name, left_side, eps_terminal_name))
            epsilon_terminals.append('empty_%s: ' % (eps_terminal_name))

    return epsilon_terminals


def _add_epsilon_for_preposition_phrases(rules_by_line):
    new_rules = []

    for left_side in rules_by_line:
        if left_side.startswith('prep_') and left_side.count('_') == 2:
            parts = left_side.split('_')
            new_rules.append(
                'empty_{phrase}: empty_prep_{prep} empty_{semtype}'.format(
                    phrase=left_side, prep=parts[1], semtype=parts[2])
            )
            new_rules.append(
                'eps_{phrase}: empty_{phrase} | {phrase}'.format(phrase=left_side))

    return new_rules


def _add_coordination_for_single_suffix(rules_by_line):
    """ Create adjective coordination for every non-terminal with suffix _single """
    coordination_terminals = []

    for left_side in rules_by_line:
        if left_side.endswith('_single'):
            coordination_terminals.append(
                """{coord}: ({single}) | (({single} ",")+ {single}) |"""
                """ (({single} ",")* {single} "a" {single})"""
                .format(single=left_side, coord=left_side[:-7])
            )

    return coordination_terminals


def _mutadd_naive_semtypes_from_combined(semantic_types):
    """ Ensure that we have all naive semantic types used in the combined ones in the grammar """
    if semantic_types is None:
        return

    for semtype in list(semantic_types.keys()):
        for naive_semtype in semtype.split('^'):
            if not naive_semtype in semantic_types:
                semantic_types[naive_semtype] = GENERATED_LINE


def _add_terminals_for_naive_semtypes(rules_by_line, semantic_types):
    """ Add terminals for each semantic types/token

    Only naive semantic types are added but they have to accept combined ones. This is the
    way how to handle ambiguity inside the grammar. Ambiguity of lemmas is not resolved here.
    The main issue is that we should generate all combinations of semantic types from USED words.
    So, processing of the data should be done twice (generate semtypes; generate grammar).

    @todo: ambiguity on the lemma level should be resolved in grammar as well
    """
    output = []

    if semantic_types is None:
        return output

    for semtype in semantic_types:
        simple_types_count = len(semtype.split('^'))

        if simple_types_count != 1:
            # do not create new terminals for combined semantic classes
            # instead add this 'string-token' into existing naive one
            continue

        terminal_semtype = semtype.upper()[1:]
        rules_by_line[terminal_semtype] = GENERATED_LINE
        tokens = []
        for semantic_type in semantic_types:
            if semtype in semantic_type:
                tokens.append(semantic_type)
        tokens.sort()
        output.append('%s: %s' % (terminal_semtype,
                                  " | ".join(['"%s"' % (s) for s in tokens])))

    return output


def _prepare_grammar(grammar):
    """ Load grammar and copy lines that will stay untouched

        :return: returns part of the grammar that is untouched and rules (with line information)
    """
    output = []
    known_rule_line = {}

    for line in grammar.split('\n'):
        line = line.strip()
        if line.startswith('//'):
            continue

        if not line:
            continue

        if not ":" in line:
            output.append(line)
            continue

        (left, right) = line.split(':', 1)
        left = left.strip()
        right = right.strip()

        if left in known_rule_line:
            output[known_rule_line[left]] += '|(%s)' % (right)
        else:
            known_rule_line[left] = len(output)
            output.append('%s:(%s)' % (left, right))

    return (output, known_rule_line)


def preprocessor(grammar, semtypes=None):
    """ Grammar preprocessor

        * allow repeating of the left-side
        * for each terminal is generated non-terminal eps_*: * | empty
    """

    (output, known_rule_line) = _prepare_grammar(grammar)

    # @note: Combined/ambiguous terminals are in format T1^T2^T3...
    # @note: naive semtype is '#foo'; combined is '#foo^#bar'
    _mutadd_naive_semtypes_from_combined(semtypes)

    output.extend(_add_terminals_for_naive_semtypes(known_rule_line, semtypes))
    output.extend(_add_epsilon_for_each_terminal(known_rule_line))
    output.extend(_add_coordination_for_single_suffix(known_rule_line))
    output.extend(_add_epsilon_for_preposition_phrases(known_rule_line))

    # add epsilon terminal
    output.append('empty:')
    # ignore white-space as a token; we are working with real-tokens separated by white-space
    output.append('%ignore " "')

    logging.debug('Expanded grammar: %s', '\n'.join(output))

    return '\n'.join(output)
