import os
import unittest
# import doctest

import numpy as np
import pandas as pd

from weelex import classifier
from weelex import lexicon
from weelex import embeddings
from weelex import trainer
from weelex.batchprocessing import batchprocessing

INPUTDIR = 'tests/testfiles/'
TEMPDIR = os.path.join(INPUTDIR, 'Temp/')


class GenericTest(unittest.TestCase):
    def _setup(self):
        self.lex1 = lexicon.Lexicon(os.path.join(INPUTDIR, 'mylex1.csv'),
                                    sep=';',
                                    encoding='latin1')
        self.lex2 = lexicon.Lexicon(os.path.join(INPUTDIR, 'mylex2.json'))

    def _setup2(self):
        self._setup()
        embeds = embeddings.Embeddings()
        embeds.load_filtered(os.path.join(TEMPDIR, 'filtered_embeddings'))
        self.embeds = embeds

    def _setup3(self):
        self._setup()
        self._setup2()
        self.lex1.embed(embeddings=self.embeds)


class MyMonkeyPatch:
    def __init__(self):
        self.colnames = ['A', 'B']
        self.df = pd.DataFrame(np.zeros((100, 2)))
        self.rnd_df = pd.DataFrame(np.random.randn(100, 2))
        self.df.columns = self.colnames
        self.rnd_df.columns = self.colnames
        self.checkpoint_path = os.path.join(TEMPDIR, 'mytest')
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
            [self.myobj.rnd_df.iloc[:50, :],
             assertion_df1.iloc[50:, :]],
             axis=0, ignore_index=True
        )
        assert x.shape == (100, 2)
        assert x.equals(assertion_df1) is False
        assert np.allclose(x, assertion_df2, rtol=0.0001)


class TestClassifier(unittest.TestCase):
    def test1(self):
        pass


class TestLexicon(GenericTest):
    def test_clean_strings(self):
        self._setup()
        array = pd.Series(
                ['Apple*', 'Banana cake * is good*', '***', '*Cucumber']
                )
        result = pd.Series(['Apple', 'Banana cake  is good', '', 'Cucumber'])
        print(result)
        print(self.lex1._clean_strings(array))
        assert self.lex1._clean_strings(array).equals(result)

    def test_copy(self):
        self._setup()
        mylex = self.lex1.copy()
        assert mylex.keys == self.lex1.keys
        self.lex1.merge(self.lex2)
        assert mylex.keys != self.lex1.keys
    
    def test_copymerge(self):
        self._setup()
        mergedlex = self.lex1.merge(self.lex2, inplace=False)
        assert mergedlex.keys != self.lex1.keys
        assert sorted(list(mergedlex.keys)) == sorted(list(self.lex1.keys) + list(self.lex2.keys))

    def test_list_padding(self):
        testlist = [1, 2, 3]
        maxlen = 5
        expected_result = [1, 2, 3, np.nan, np.nan]
        assert lexicon.list_padding(testlist, maxlen=maxlen) == expected_result

    def test_dict_padding(self):
        testdict = {'A': [1, 2, 3], 'B': [1, 2, 3, 4, 5]}
        expected_result = {'A': [1, 2, 3, np.nan, np.nan], 'B': [1, 2, 3, 4, 5]}
        assert lexicon.dict_padding(testdict) == expected_result

    def test_build_csv(self):
        self._setup()
        assert self.lex1.keys == ['PolitikVR', 'AutoVR']
        assert self.lex1._dictionary_df.shape == (55, 2)

    def test_build_json(self):
        self._setup()
        assert self.lex2.keys == ['Space', 'Food']
        assert self.lex2._dictionary_df.shape == (9, 2)

    def test_keys(self):
        self._setup()
        assert isinstance(self.lex1.keys, list)
        assert isinstance(self.lex2.keys, list)

    def test_merge1(self):
        self._setup()
        self.lex1.merge(self.lex2)
        assert self.lex1.keys == ['PolitikVR', 'AutoVR', 'Space', 'Food']
        assert self.lex1._dictionary_df.shape == (55, 4)

    def test_merge2(self):
        self._setup()
        self.lex2.merge(self.lex1)
        assert self.lex2.keys == ['Space', 'Food', 'PolitikVR', 'AutoVR', ]
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
        assert sorted(dct.keys()) == sorted(self.lex1.keys)

    def test_properties(self):
        self._setup()
        assert self.lex1.get_vocabulary() == self.lex1.vocabulary
        assert isinstance(self.lex1.keys, list)

    def test_embedding(self):
        self._setup2()
        self.lex1.embed(self.embeds)
        assert self.lex1.embeddings is not None
        assert isinstance(self.lex1.embeddings, np.ndarray)

    def test_embedding_shape(self):
        self._setup2()
        self.lex1.embed(self.embeds)
        assert len(self.lex1.embedding_shape) == 3
        assert self.lex1.embedding_shape[2] == 300
        assert self.lex1.embedding_shape[1] == len(self.lex1.keys)
        assert self.lex1.embedding_shape[0] == len(self.lex1[self.lex1.keys[0]])

    def test_getter(self):
        self._setup()
        keys = self.lex1.keys
        assert isinstance(self.lex1[keys[0]], pd.Series)


class TestEmbeddings(unittest.TestCase):
    def _setup(self):
        self.lex1 = lexicon.Lexicon(os.path.join(INPUTDIR, 'mylex1.csv'),
                                    sep=';', 
                                    encoding='latin1')
        self.lex2 = lexicon.Lexicon(os.path.join(INPUTDIR, 'mylex2.json'))
        self.lex1.merge(self.lex2)
        self.vocab = self.lex1.get_vocabulary()
        self.path = os.path.join(TEMPDIR, 'filtered_embeddings')

    def _setup2(self):
        self._setup()
        embeds = embeddings.Embeddings()
        embeds.load_filtered(self.path)
        self.embeds = embeds

    def test_data_from_dict(self):
        x = np.random.randn(10)
        y = np.random.randn(10)
        my_dct = {'A': x, 'B': y}
        embeds = embeddings.Embeddings()
        keys, vectors = embeds._data_from_dct(my_dct)
        assert vectors.shape == (2, 10)
        assert np.allclose(vectors[0], x)
        assert np.allclose(vectors[1], y)
        assert list(keys) == ['A', 'B']

    def test_load_filtered(self):
        self._setup()
        embeds = embeddings.Embeddings()
        embeds.load_filtered(self.path)
        assert isinstance(embeds.keys, np.ndarray)
        assert isinstance(embeds._vectors, np.ndarray)
        assert embeds._vectors.shape[1] == 300
        assert sorted(list(embeds.keys)) == sorted(list(self.vocab))

    # def test_lookup2(self):
    #     self._setup()
    #     terms = [self.vocab[i] for i in [1,3,5]]
    #     embeds = embeddings.Embeddings()
    #     embeds.load_filtered(self.path)
    #     v1 = embeds.lookup2(terms[0])
    #     v2 = embeds[terms[0]]
    #     v3 = embeds.lookup2(terms)
    #     v4 = embeds.lookup2(np.array(terms))
    #     assert isinstance(v1, np.ndarray)
    #     assert isinstance(v2, np.ndarray)
    #     assert np.allclose(v1, v2)
    #     assert isinstance(v3, np.ndarray)
    #     assert isinstance(v4, np.ndarray)
    #     assert v3.shape == (3, 300)
    #     assert v4.shape == (3, 300)

    def test_lookup(self):
        self._setup2()
        terms = [self.vocab[i] for i in [1, 3, 5]]
        v1 = self.embeds.lookup(terms[0])
        v2 = self.embeds[terms[0]]
        v3 = self.embeds.lookup(terms)
        assert isinstance(v1, np.ndarray)
        assert isinstance(v2, np.ndarray)
        assert np.allclose(v1, v2)
        assert isinstance(v3, np.ndarray)
        assert v3.shape == (3, 300)

    def test_properties(self):
        self._setup2()
        assert self.embeds.dim == 300
        assert isinstance(self.embeds.keys, np.ndarray)


class TestTrainer(GenericTest):
    def _setup4(self):
        self._setup3()
        self.tr = trainer.TrainProcessor(lex=self.lex1,
                                         support_lex=self.lex2,
                                         embeddings=self.embeds)

    def _setup5(self):
        self._setup4()
        self.tr._prepare_inputs()

    def test(self):
        pass

    def test_embedding_df(self):
        self._setup4()
        df = self.tr._get_embedding_df(self.lex1)
        assert isinstance(df, pd.DataFrame)
        assert len( df.shape ) == 2
        assert df.shape[1] == 300

    def test_term_id_mapping(self):
        self._setup4()
        df = self.tr._get_embedding_df(self.lex1)
        term2cat = self.tr._make_id_term_mapping(df)
        assert isinstance(term2cat, pd.DataFrame) 
        assert len(term2cat.shape) == 2
        assert term2cat.shape[1] == 2
        assert list(term2cat.columns) == ['categories', 'terms']

    def test_make_y(self):
        self._setup5()
        y = self.tr._make_y()
        print(y)
        assert isinstance(y, dict)
        print(y.keys(), self.lex1.keys)
        assert sorted(list(y.keys())) == sorted(list(self.lex1.keys ))
        
    def test_make_x(self):
        self._setup5()
        y = self.tr._make_y()
        x = self.tr._make_X(y)
        assert isinstance(x, dict)
        assert sorted(list(y.keys())) == sorted(list(self.lex1.keys))
        
    def test_make_data(self):
        self._setup5()
        x, y = self.tr.transform()
        assert isinstance(x, dict)
        assert isinstance(y, dict)

if __name__ == '__main__':
    unittest.main()
