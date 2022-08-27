import os
import unittest

import numpy as np

from weelex import classifier
from weelex import lexicon
from weelex import embeddings

class TestClassifier(unittest.TestCase):
    def test1(self):
        pass


class TestLexicon(unittest.TestCase):
    def test_list_padding(self):
        testlist = [1,2,3]
        maxlen = 5
        expected_result = [1,2,3, np.nan, np.nan]
        assert lexicon.list_padding(testlist, maxlen=maxlen) == expected_result

    def test_dict_padding(self):
        testdict = {'A': [1,2,3], 'B': [1,2,3,4,5]}
        expected_result = {'A': [1,2,3,np.nan,np.nan], 'B': [1,2,3,4,5]}
        assert lexicon.dict_padding(testdict) == expected_result


class TestEmbeddings(unittest.TestCase):
    def test1(self):
        pass



if __name__=='__main__':
    unittest.main()
