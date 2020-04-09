import unittest

from pipeline import add_semtypes_for_lemma
from preprocessor import preprocessor


class TestAddSemtypeForLemma(unittest.TestCase):
    def test_naive(self):
        vocabulary = {
            'foo': {'#foo'}
        }

        self.assertEqual(
            add_semtypes_for_lemma(vocabulary, 'foo'),
            ['#foo']
        )

    def test_combined(self):
        vocabulary = {
            'foo': {'#foo', '#bar'}
        }

        self.assertEqual(
            add_semtypes_for_lemma(vocabulary, 'foo'),
            ['#bar^#foo']
        )

    def test_combined_with_floskule(self):
        vocabulary = {
            'foo': {'#foo', '#floskule', '#bar'}
        }

        self.assertEqual(
            add_semtypes_for_lemma(vocabulary, 'foo'),
            ['#bar^#foo', '#floskule']
        )


class TestPreprocessor(unittest.TestCase):
    # @todo: create own assertEqual that will call preprocessor and adds self.permanent_suffix
    permanent_suffix = '\nempty:\n%ignore " "'

    def test_comments_are_removed(self):
        grammar = "// this is comment line"
        self.assertEqual("\n" + preprocessor(grammar),
                         "" + self.permanent_suffix)

    def test_commands_are_left_untouched(self):
        grammar = "\%ignore abc"
        self.assertEqual(preprocessor(grammar),
                         '\%ignore abc' + self.permanent_suffix)

    def test_empty_lines_are_removed(self):
        grammar = """


        """
        self.assertEqual("\n" + preprocessor(grammar),
                         "" + self.permanent_suffix)

    def test_order_of_rules_is_untouched(self):
        """ Test if the content of rule is untouched """
        grammar = """
        sentence: foo
        foo: bar
        """
        self.assertEqual(preprocessor(grammar).strip(),
                         'sentence:(foo)\nfoo:(bar)' + self.permanent_suffix)

    def test_parentheses_are_added_to_right_side(self):
        """ Test if the parentheses are added to the right side of the rule """
        grammar = "   sentence: foo  "
        self.assertEqual(preprocessor(grammar).strip(),
                         'sentence:(foo)' + self.permanent_suffix)

    def test_merge_of_two_left_sides(self):
        """ Test if the left sides are merged if they are across multiple lines """
        grammar = """
            sentence: foo
            sentence: bar
        """
        self.assertEqual(preprocessor(grammar).strip(),
                         'sentence:(foo)|(bar)' + self.permanent_suffix)

    def test_merge_of_two_left_sides_with_inserted_lines(self):
        grammar = """
            sentence: foo
            // this is my comment
            foo: bar
            sentence: bar
        """
        self.assertEqual(preprocessor(grammar).strip(),
                         'sentence:(foo)|(bar)\nfoo:(bar)' + self.permanent_suffix)

    def test_if_epsilon_nonterminal_was_added_for_terminal(self):
        grammar1 = """
            TERMINAL: "foo"
        """
        self.assertEqual(preprocessor(grammar1).strip(),
                         'TERMINAL:("foo")\neps_terminal: TERMINAL | empty' + self.permanent_suffix)

        grammar2 = """
            TERMINAL: "foo"
            TERMINAL: "bar"
        """
        self.assertEqual(preprocessor(grammar2).strip(),
                         'TERMINAL:("foo")|("bar")\neps_terminal: TERMINAL | empty' + self.permanent_suffix)

    def test_if_coordination_is_added_for_single_suffix(self):
        grammar = """
        t_attr_single: t_quality* ATTR
        """
        self.assertEqual(preprocessor(
            grammar), "t_attr_single:(t_quality* ATTR)\nt_attr: (t_attr_single) | ((t_attr_single \",\")+ t_attr_single) | ((t_attr_single \",\")* t_attr_single \"a\" t_attr_single)" + self.permanent_suffix)

    def test_generate_simple_terminals(self):
        grammar = ""
        self.assertEqual(preprocessor(
            grammar, {'#foo': 1}), 'FOO: "#foo"\neps_foo: FOO | empty' + self.permanent_suffix)

    def test_generate_merged_terminals(self):
        semtypes = {'#floskule^#measure': 1, '#floskule': 1, '#measure': 1}
        grammar = ""

        self.assertEqual(preprocessor(grammar, semtypes),
                         """FLOSKULE: "#floskule" | "#floskule^#measure"
MEASURE: "#floskule^#measure" | "#measure"
eps_floskule: FLOSKULE | empty
eps_measure: MEASURE | empty""" + self.permanent_suffix)

    def test_generate_merged_terminals_wo_naives(self):
        semtypes = {'#floskule^#measure': 1}
        grammar = ""

        self.assertEqual(preprocessor(grammar, semtypes),
                         """FLOSKULE: "#floskule" | "#floskule^#measure"
MEASURE: "#floskule^#measure" | "#measure"
eps_floskule: FLOSKULE | empty
eps_measure: MEASURE | empty""" + self.permanent_suffix)


if __name__ == '__main__':
    unittest.main()
