import unittest

from pipeline import preprocessor


class TestPreprocessor(unittest.TestCase):
    def test_comments_are_removed(self):
        grammar = "// this is comment line"
        self.assertEqual(preprocessor(grammar), "")

    def test_commands_are_left_untouched(self):
        grammar = "\%ignore abc"
        self.assertEqual(preprocessor(grammar), '\%ignore abc')

    def test_empty_lines_are_removed(self):
        grammar = """


        """
        self.assertEqual(preprocessor(grammar), "")

    def test_order_of_rules_is_untouched(self):
        """ Test if the content of rule is untouched """
        grammar = """
        sentence: foo
        foo: bar
        """
        self.assertEqual(preprocessor(grammar).strip(),
                         'sentence:(foo)\nfoo:(bar)')

    def test_parentheses_are_added_to_right_side(self):
        """ Test if the parentheses are added to the right side of the rule """
        grammar = "   sentence: foo  "
        self.assertEqual(preprocessor(grammar).strip(), 'sentence:(foo)')

    def test_merge_of_two_left_sides(self):
        """ Test if the left sides are merged if they are across multiple lines """
        grammar = """
            sentence: foo
            sentence: bar
        """
        self.assertEqual(preprocessor(grammar).strip(), 'sentence:(foo)|(bar)')

    def test_merge_of_two_left_sides_with_inserted_lines(self):
        grammar = """
            sentence: foo
            // this is my comment
            foo: bar
            sentence: bar
        """
        self.assertEqual(preprocessor(grammar).strip(),
                         'sentence:(foo)|(bar)\nfoo:(bar)')


if __name__ == '__main__':
    unittest.main()
