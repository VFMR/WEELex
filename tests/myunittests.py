import os
import unittest

import numpy as np
import pandas as pd

from weelex import classifier
from weelex import lexicon
from weelex import embeddings
from weelex import batchprocessing

INPUTDIR = 'tests/testfiles/'
TEMPDIR = os.path.join(INPUTDIR, 'Temp/')


class MyMonkeyPatch:
    def __init__(self):
        self.colnames = ['A', 'B']
        self.df = pd.DataFrame(np.zeros((100, 2)))
        self.rnd_df = pd.DataFrame(np.random.randn(100,2))
        self.df.columns = self.colnames
        self.rnd_df.columns = self.colnames
        self.checkpoint_path=os.path.join(TEMPDIR, 'mytest')
        self.fake_cp_path = os.path.join(TEMPDIR, 'mytest_fake')

    @batchprocessing.batch_predict
    def add(self, X, n_batches=None, checkpoint_path=None):
        return X + 1


class TestBatchProc(unittest.TestCase):
    def _setup(self):
        self.myobj = MyMonkeyPatch()
        batchprocessing.check_makedir(self.myobj.fake_cp_path)
        splits = np.array_split(self.myobj.rnd_df, 10)
        for i in range(5):
            splits_df = pd.DataFrame(splits[i])
            batchprocessing.save_checkpoints(
                self.myobj.fake_cp_path,
                iteration=i,
                df=splits_df
            )

    def test_batch_predict(self):
        self._setup()
        x = self.myobj.add(X=self.myobj.df,
                           n_batches=10,
                           checkpoint_path=self.myobj.checkpoint_path)
        assertion_df = pd.DataFrame(np.ones((100, 2)))
        assertion_df.columns = self.myobj.colnames
        assert x.shape == (100, 2)
        assert x.equals(assertion_df)

    def test_check_makedir(self):
        self._setup()
        batchprocessing.check_makedir(self.myobj.checkpoint_path)
        assert os.path.isdir(self.myobj.checkpoint_path)

    def test_cleanup_checkpoints(self):
        self._setup()
        batchprocessing.check_makedir(self.myobj.checkpoint_path)
        batchprocessing.cleanup_checkpoints(self.myobj.checkpoint_path)
        assert os.path.isdir(self.myobj.checkpoint_path) is False

    def test_load_checkpoints(self):
        self._setup()
        x = self.myobj.add(X=self.myobj.df,
                           n_batches=10,
                           checkpoint_path=self.myobj.fake_cp_path)
        assertion_df1 = pd.DataFrame(np.ones((100, 2)))
        assertion_df1.columns = self.myobj.colnames
        assertion_df2 = pd.concat(
            [self.myobj.rnd_df.iloc[:50,:],
            assertion_df1.iloc[50:,:]],
            axis=0, ignore_index=True
        )
        assert x.shape == (100, 2)
        assert x.equals(assertion_df1) is False
        assert np.allclose(x, assertion_df2, rtol=0.0001)


class TestClassifier(unittest.TestCase):
    def test1(self):
        pass


class TestLexicon(unittest.TestCase):
    def _setup(self):
        self.lex1 = lexicon.Lexicon(os.path.join(INPUTDIR, 'mylex1.csv'), sep=';')
        self.lex2 = lexicon.Lexicon(os.path.join(INPUTDIR, 'mylex2.json'))

    def test_list_padding(self):
        testlist = [1,2,3]
        maxlen = 5
        expected_result = [1,2,3, np.nan, np.nan]
        assert lexicon.list_padding(testlist, maxlen=maxlen) == expected_result

    def test_dict_padding(self):
        testdict = {'A': [1,2,3], 'B': [1,2,3,4,5]}
        expected_result = {'A': [1,2,3,np.nan,np.nan], 'B': [1,2,3,4,5]}
        assert lexicon.dict_padding(testdict) == expected_result

    def test_build_csv(self):
        self._setup()
        assert self.lex1.keys() == ['PolitikVR', 'AutoVR']
        assert self.lex1._dictionary_df.shape == (55, 2)

    def test_build_json(self):
        self._setup()
        assert self.lex2.keys() == ['Space', 'Food']
        assert self.lex2._dictionary_df.shape == (9, 2)

    def test_merge1(self):
        self._setup()
        self.lex1.merge(self.lex2)
        assert self.lex1.keys() == ['PolitikVR', 'AutoVR', 'Space', 'Food']
        assert self.lex1._dictionary_df.shape == (55, 4)

    def test_merge2(self):
        self._setup()
        self.lex2.merge(self.lex1)
        assert self.lex2.keys() == ['Space', 'Food', 'PolitikVR', 'AutoVR', ]
        assert self.lex2._dictionary_df.shape == (55, 4)

    def test_get_vocabulary(self):
        self._setup()
        vocab = self.lex1.get_vocabulary()
        assert isinstance(vocab, list)
        assert isinstance(vocab[0], str)

    def test_to_dict(self):
        self._setup()
        dct = self.lex1.to_dict()
        assert isinstance(dct, dict)
        assert sorted(dct.keys()) == sorted(self.lex1.keys())


class TestEmbeddings(unittest.TestCase):
    def test1(self):
        pass



if __name__=='__main__':
    unittest.main()
