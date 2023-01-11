import os
import unittest
# import doctest

import numpy as np
import pandas as pd

from weelex import classifier
from weelex import lexicon
from weelex import embeddings
from weelex import trainer
from weelex import ensemble
from weelex import predictor
from weelex import lsx
from batchprocessing import batchprocessing
from cluster_tfidf import cluster_tfidf

INPUTDIR = 'tests/testfiles/'
TEMPDIR = os.path.join(INPUTDIR, 'Temp/')


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


class GenericTest(unittest.TestCase):
    def _setup(self):
        self.lex1 = lexicon.Lexicon(os.path.join(INPUTDIR, 'mylex1.csv'),
                                    sep=';',
                                    encoding='latin1')
        self.lex2 = lexicon.Lexicon(os.path.join(INPUTDIR, 'mylex2.json'))

    def _setup2(self):
        self._setup()
        embeds = embeddings.Embeddings.load_filtered(os.path.join(TEMPDIR, 'filtered_embeddings'))
        self.embeds = embeds

    def _setup3(self):
        self._setup2()
        self.lex1.embed(embeddings=self.embeds)

    def _setup_predictor(self):
        self._setup2()
        data = pd.Series(
            [
            'Ich esse gerne Kuchen und andere Süßigkeiten',
            'Dort steht ein schnelles Auto mit einem Lenkrad und Reifen.',
            'Die Politik von heute ist nicht mehr die gleiche wie damals.',
            'Hier ist nochmal ein sehr generischer Satz.',
            'Wie ist das Wetter heute?',
            'Ich esse gerne Kuchen und andere Süßigkeiten',
            'Dort steht ein schnelles Auto mit einem Lenkrad und Reifen.',
            'Die Politik von heute ist nicht mehr die gleiche wie damals.',
            'Hier ist nochmal ein sehr generischer Satz.',
            'Wie ist das Wetter heute?',
            'Ich esse gerne Kuchen und andere Süßigkeiten',
            'Dort steht ein schnelles Auto mit einem Lenkrad und Reifen.',
            'Die Politik von heute ist nicht mehr die gleiche wie damals.',
            'Hier ist nochmal ein sehr generischer Satz.',
            'Wie ist das Wetter heute?',
            'Ich esse gerne Kuchen und andere Süßigkeiten',
            'Dort steht ein schnelles Auto mit einem Lenkrad und Reifen.',
            'Die Politik von heute ist nicht mehr die gleiche wie damals.',
            'Hier ist nochmal ein sehr generischer Satz.',
            'Wie ist das Wetter heute?',
            ])
        self.processor = predictor.PredictionProcessor(
            data = data,
            embeddings=self.embeds,
            tfidf=None,
            ctfidf=None,
            min_df=1,
            max_df=0.99,
            n_docs=len(data),
            n_words=10,
        )
        self.data = data


class TestBatchProc(unittest.TestCase):
    def _setup(self):
        self.myobj = MyMonkeyPatch()
        batchprocessing._check_makedir(self.myobj.fake_cp_path)
        splits = np.array_split(self.myobj.rnd_df, 10)
        for i in range(5):
            splits_df = pd.DataFrame(splits[i])
            batchprocessing._save_checkpoints(
                self.myobj.fake_cp_path,
                iteration=i,
                df=splits_df,
                n_batches=10,
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
        batchprocessing._check_makedir(self.myobj.checkpoint_path)
        assert os.path.isdir(self.myobj.checkpoint_path)

    def test_cleanup_checkpoints(self):
        self._setup()
        batchprocessing._check_makedir(self.myobj.checkpoint_path)
        batchprocessing._cleanup_checkpoints(self.myobj.checkpoint_path)
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


class TestPredictor(GenericTest):
    def test1(self):
        self._setup_predictor()
        assert isinstance(self.processor, predictor.PredictionProcessor)

        # Test tfidf-fitting
        self.processor.fit_tfidf()
        assert self.processor._tfidf.check_fit()
        print(self.processor._tfidf.vectorizer[-1].vocabulary_)

        # test ctfidf-fitting:
        print(self.processor._tfidf)
        self.processor.fit_ctfidf()
        assert True

        # test transformation with ctfidf:
        vects = self.processor.transform()
        print(vects)
        assert isinstance(vects, np.ndarray)
        assert vects.shape == (20, 300)

        vects = self.processor.transform(self.data)
        print(vects)
        assert isinstance(vects, np.ndarray)
        assert vects.shape == (20, 300)

        # test transformation without ctfidf:
        self.processor._use_ctfidf = False
        vects = self.processor.transform()
        print(vects)
        assert isinstance(vects, np.ndarray)
        assert vects.shape == (20, 300)



class TestClassifier(GenericTest):
    def test_setup(self):
        self._setup2()
        cl = classifier.WEELexClassifier(embeds=self.embeds)


    def test_fit(self):
        self._setup_predictor()
        cl = classifier.WEELexClassifier(embeds=self.embeds)
        cl.fit(X=self.data,lex=self.lex1,
                     support_lex=self.lex2,
                     hp_tuning=False)
        assert cl._is_fit is True
        assert isinstance(cl._models, dict)
        assert len(cl._models) > 0
        assert cl.main_keys == ['PolitikVR', 'AutoVR']
        assert cl.support_keys == ['Space', 'Food']

        X, y = cl._trainprocessor.feed_cat_Xy(cat='PolitikVR', train=True)
        X, y = cl._trainprocessor.feed_cat_Xy(cat='AutoVR', train=True)
        assert len(X) == len(y)

    def test_fit_pb(self):
        self._setup_predictor()
        cl = classifier.WEELexClassifier(embeds=self.embeds)
        cl.fit(X=self.data,
               lex=self.lex1,
                     support_lex=self.lex2,
                     hp_tuning=False,
                     progress_bar=True)
        assert cl._is_fit is True
        assert isinstance(cl._models, dict)
        assert len(cl._models) > 0
        assert cl.main_keys == ['PolitikVR', 'AutoVR']

    def test_hp_tuning(self):
        self._setup_predictor()
        n_best_params = 4
        cl = classifier.WEELexClassifier(
            embeds=self.embeds,
            test_size=0.5,
            random_state=1)
        param_grid = [{'modeltype': ['svm'],
                       'n_models': [2],
                       'pca': [10, None],
                       'svc_c': [0.1, 1, 10]}]
        cl.fit(X=self.data,
               lex=self.lex1,
               support_lex=self.lex2,
               hp_tuning=True,
               param_grid=param_grid,
               progress_bar=True,
               n_iter=6,
               n_best_params=n_best_params,
               cv=3)
        assert cl._is_fit is True
        assert isinstance(cl._models, dict)
        assert len(cl._models) > 0
        assert cl.main_keys == ['PolitikVR', 'AutoVR']
        print(cl._cv_scores)
        print(cl._tuned_params)
        assert list(cl._tuned_params.keys()) == ['PolitikVR', 'AutoVR']
        assert len(cl._tuned_params['AutoVR']) == n_best_params

    def test_weelexpredict(self):
        self._setup_predictor()
        n_best_params = 4
        cl = classifier.WEELexClassifier(
            embeds=self.embeds,
            relevant_pos=['NOUN'],
            min_df=1,
            max_df=0.99,
            n_docs=len(self.data),
            n_words=10,
        )
        param_grid = [{'modeltype': ['svm'],
                       'n_models': [2],
                       'pca': [10, None],
                       'svc_c': [0.1, 1, 10]}]
        cl.fit(X=self.data,
                lex=self.lex1,
                     support_lex=self.lex2,
                     hp_tuning=True,
                     param_grid=param_grid,
                     progress_bar=True,
                     n_iter=6,
                     n_best_params=n_best_params,
                     cv=3)
        preds = cl.predict_docs(X=self.data)
        print(preds)
        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == (20, 2)
        assert list(preds.columns) == self.lex1.keys
        assert list(preds['PolitikVR'].unique()) == [0, 1]

        # without hp tuning:
        cl = classifier.WEELexClassifier(
            embeds=self.embeds,
            relevant_pos=['NOUN'],
            min_df=1,
            max_df=0.99,
            n_docs=len(self.data),
            n_words=10,
        )
        cl.fit(X=self.data,
                 lex=self.lex1,
                     support_lex=self.lex2,
                     hp_tuning=False,
                     progress_bar=True,
                     n_iter=6,
                     n_best_params=n_best_params,
                     cv=3)
        preds = cl.predict_docs(X=self.data)
        print(preds)
        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == (20, 2)
        assert list(preds.columns) == self.lex1.keys
        assert list(preds['PolitikVR'].unique()) == [0, 1]

        # without ctfidf
        cl = classifier.WEELexClassifier(
            embeds=self.embeds,
            use_ctfidf=False,
            relevant_pos=['NOUN'],
            min_df=1,
            max_df=0.99,
            n_docs=len(self.data),
            n_words=10,
        )
        cl.fit(X=self.data,
                lex=self.lex1,
                     support_lex=self.lex2,
                     hp_tuning=False,
                     progress_bar=True,
                     n_iter=6,
                     n_best_params=n_best_params,
                     cv=3)
        preds = cl.predict_docs(X=self.data)
        print(preds)
        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == (20, 2)
        assert list(preds.columns) == self.lex1.keys
        assert list(preds['PolitikVR'].unique()) == [0, 1]


class TestModels(GenericTest):
    def _setup_augmented(self):
        self._setup2()
        self.cat = 'PolitikVR'
        self.categories = self.lex1.keys
        self.support_categories = self.lex2.keys
        # self.lex1.merge(self.lex2, inplace=True)
        self.lex1.embed(embeddings=self.embeds)
        self.data = self.lex1.embeddings

    def _setup_augmented2(self):
        self._setup_augmented()
        model = ensemble.AugmentedEnsemble(category=self.cat,
                                  categories=self.categories,
                                  outside_categories=self.support_categories)
        self.model = model

    def test_get_targets(self):
        self._setup_augmented2()
        tt = [1,0,0,1,1,1,0,1,0,0]
        y = pd.DataFrame({'col1': tt,
                          'col2': [0,0,0,0,0,0,0,0,0,0]})
        targets = self.model._get_targets(np.array(y))
        assert targets.shape == (10,)
        assert pd.DataFrame(targets).equals(pd.DataFrame(tt))

    def test_data_input(self):
        self._setup_augmented2()
        tt = [1,0,0,1,1,1,0,1,0,1]
        y = pd.DataFrame({'col1': tt,
                          'col2': [0,0,0,0,0,0,0,0,0,0]})
        targets = self.model._get_targets(np.array(y))

        # numpy array:
        X_np = np.random.randn(len(tt), 5)
        y_np = np.array(y.copy())
        new_X1 = self.model._get_drawclass_data(X_np, targets, classvalue=1)
        new_X0 = self.model._get_drawclass_data(X_np, targets, classvalue=0)
        new_y1 = self.model._get_drawclass_data(y_np, targets, classvalue=1)
        new_y0 = self.model._get_drawclass_data(y_np, targets, classvalue=0)
        assert len(new_X0)==4
        assert len(new_X1)==6
        assert len(new_y1) == len(new_X1)
        assert len(new_y0) == len(new_X0)

        # pandas df
        X_df = pd.DataFrame(X_np)
        y_df = pd.DataFrame(y_np)
        new_X1 = self.model._get_drawclass_data(X_df, targets, classvalue=1)
        new_X0 = self.model._get_drawclass_data(X_df, targets, classvalue=0)
        new_y1 = self.model._get_drawclass_data(y_df, targets, classvalue=1)
        new_y0 = self.model._get_drawclass_data(y_df, targets, classvalue=0)
        assert len(new_X0)==4
        assert len(new_X1)==6
        assert len(new_y1) == len(new_X1)
        assert len(new_y0) == len(new_X0)

    def test_random_category(self):
        self._setup_augmented2()
        oc = self.model.outside_categories_all
        cc = self.model.category
        rc = self.model._random_category(oc,cc)
        assert rc in self.model.outside_categories_all
        assert isinstance(rc, str)

    def test_getkeeps(self):
        self._setup_augmented2()
        tt = [True,False,False,True,True,True,False,True,False,True]
        y = pd.DataFrame({'col1': tt,
                          'col2': ['PolitikVR','Space','AutoVR','PolitikVR','PolitikVR','PolitikVR','Space','PolitikVR','Food','PolitikVR']})
        X = np.random.randn(len(tt), 5)
        print('category:', self.model.category)
        keep1 = self.model._getkeeps(X, y, classvalue=1)
        print(y)
        print('category:', self.model.category)
        print('outside_categories: ', self.model.outside_categories_all)
        assert isinstance(self.model.category, str)
        keep0 = self.model._getkeeps(X, y, classvalue=0)
        spacekeep = [False, True, False, False, False, False, True, False, False, False]
        foodkeep = [False, False, False, False, False, False, False, False, True, False]
        autokeep = [False, False, True, False, False, False, False, False, False, False]
        allkeep = [True, True, True, True, True, True, True, True, True, True]
        assert isinstance(keep1, list)
        assert len(keep1) == len(tt)
        assert len(keep0) == len(tt)
        assert keep1 == allkeep
        print(keep0)
        assert keep0 == spacekeep or keep0==foodkeep or keep0==autokeep


    def test_make_agg_sample(self):
        shape = (100, 10)
        array = np.random.randn(*shape)
        n = 3
        new_array = ensemble.make_agg_sample(X=array, n=n)
        assert new_array.shape == (shape[1],)

    def test(self):
        self._setup_augmented2()
        # self.model.draw_random_samples_classwise()

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
        assert lexicon._list_padding(testlist, maxlen=maxlen) == expected_result

    def test_dict_padding(self):
        testdict = {'A': [1, 2, 3], 'B': [1, 2, 3, 4, 5]}
        expected_result = {'A': [1, 2, 3, np.nan, np.nan], 'B': [1, 2, 3, 4, 5]}
        assert lexicon._dict_padding(testdict) == expected_result

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
        embeds = embeddings.Embeddings.load_filtered(self.path)
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
        embeds = embeddings.Embeddings.load_filtered(self.path)
        assert isinstance(embeds.keys, np.ndarray)
        assert isinstance(embeds._vectors, np.ndarray)
        assert embeds._vectors.shape[1] == 300
        assert sorted(list(embeds.keys)) == sorted(list(set(
            list(self.vocab) + [
                'super', 'gut', 'furchtbar', 'schlecht', 'wunderschön',
                'hässlich', 'mies', 'lecker', 'Kuchen', 'Lenkrad', 'Reifen',
                'Satz', 'fantastisch', 'schön', 'blöd', 'Wetter', 'generisch'])))

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

class TestWeightedLex(GenericTest):
    def _setup_weighted(self):
        self.mydct1 = {'super': 0.9, 'gut': 0.7, 'schlecht': -0.7, 'furchtbar': -0.9}
        self.mydct2 = pd.DataFrame([['super', 0.9],
                                    ['gut', 0.7],
                                    ['schlecht', -0.7],
                                    ['furchtbar', -0.9]])
        self.mydct3 = pd.DataFrame({0: ['super', 'gut', 'schlecht', 'furchtbar'],
                                    1: [0.9, 0.7, -0.7, -0.9]})

    def _setup_weighted2(self):
        self._setup_weighted()
        self._setup2()  # load embeddings
        self.lex = lexicon.WeightedLexicon(self.mydct1)

    def test_input(self):
        self._setup_weighted()
        lex1 = lexicon.WeightedLexicon(self.mydct1)
        lex2 = lexicon.WeightedLexicon(self.mydct2)
        lex3 = lexicon.WeightedLexicon(self.mydct3)
        assert lex1.vocabulary == lex2.vocabulary
        assert lex2.vocabulary == lex3.vocabulary
        assert list(lex1.weights) == list(lex2.weights)
        assert list(lex2.weights) == list(lex3.weights)
        print(lex1._dictionary_df)
        print(lex1.weights)
        assert sorted(lex1.vocabulary) == sorted(['super', 'gut', 'schlecht', 'furchtbar'])
        assert list(lex1.weights) == [0.9, 0.7, -0.7, -0.9]

    def test_embedding(self):
        self._setup_weighted2()
        self.lex.embed(self.embeds)
        print(self.lex.embeddings)
        print(self.lex.embeddings.shape)
        assert self.lex.embeddings.shape == (4,300)
        assert np.allclose(self.embeds['super'], self.lex.embeddings[0,:])
        assert np.allclose(self.embeds['gut'], self.lex.embeddings[1,:])
        assert np.allclose(self.embeds['schlecht'], self.lex.embeddings[2,:])
        assert np.allclose(self.embeds['furchtbar'], self.lex.embeddings[3,:])


class TestLSX(GenericTest):
    def _setup_lsx(self):
        self._setup2()
        data = pd.Series(
            [
            'Kuchen finde ich super, weil er gut ist.',
            'Dort steht ein schlechtes Auto mit einem Lenkrad und Reifen.',
            'Die Politik von heute ist nicht mehr die gleiche wie damals.',
            'Hier ist nochmal ein wunderschöner generischer Satz.',
            'Wie ist das Wetter heute? Es ist mies!',
            'Kuchen und Süßigkeiten finde ich furchtbar',
            'Dort steht ein gutes Auto mit einem Lenkrad und Reifen.',
            'Die Politik von heute ist nicht mehr die gleiche wie damals.',
            'Hier ist nochmal ein wunderschöner generischer Satz.',
            'Wie ist das Wetter heute? Es ist mies!',
            'Kuchen finde ich sehr lecker',
            'Dort steht ein hässliches Auto mit einem Lenkrad und Reifen.',
            'Die Politik von heute ist nicht mehr die gleiche wie damals.',
            'Hier ist nochmal ein wunderschöner generischer Satz.',
            'Wie ist das Wetter heute? Es ist mies!',
            'Kuchen finde ich sehr lecker',
            'Dort steht ein hässliches Auto mit einem Lenkrad und Reifen.',
            'Die Politik von heute ist nicht mehr die gleiche wie damals.',
            'Hier ist nochmal ein wunderschöner generischer Satz.',
            'Wie ist das Wetter heute? Es ist mies!',
            'Kuchen finde ich sehr lecker',
            'Dort steht ein hässliches Auto mit einem Lenkrad und Reifen.',
            'Die Politik von heute ist nicht mehr die gleiche wie damals.',
            'Hier ist nochmal ein wunderschöner generischer Satz.',
            'Wie ist das Wetter heute? Es ist mies!',
            ])
        self.processor = predictor.PredictionProcessor(
            data = data,
            embeddings=self.embeds,
            tfidf=None,
            ctfidf=None,
            relevant_pos=['ADJ', 'ADV', 'NOUN', 'VERB'],
            min_df=1,
            max_df=0.99,
            n_docs=len(data),
            n_words=10,
        )
        self.data = data

        self.mydct1 = {'super': 0.9, 'gut': 0.7, 'schlecht': -0.7, 'furchtbar': -0.9}
        self.lex = lexicon.WeightedLexicon(self.mydct1)
        self.model = lsx.LatentSemanticScaling(embeds=self.embeds)

    def test_cosine(self):
        a = [1,0,0]
        b = [0,1,0]
        assert lsx._cosine_simil(a, b) == 0.0

        a = [1,0,0]
        b = [1,0,0]
        assert lsx._cosine_simil(a, b) == 1.0

        a = [1, 0, 0]
        b = [1, 1, 1]
        assert np.allclose(lsx._cosine_simil(a, b), 0.5773502691896257)

    def test_lsx(self):
        self._setup_lsx()
        self.model.fit(polarity_lexicon=self.lex, X=self.data)
        vectors = self.processor.transform(self.data)
        # print(vectors)
        # print(vectors.shape)
        # print('embedding_shape: ', self.lex.embeddings.shape)
        print('embedding_shape-row: ', self.lex.embeddings[0,:].shape)
        preds = self.model.predict_docs(self.data)
        print(preds)
        polarity1 = self.model.polarity('gut')
        polarity2 = self.model.polarity('schlecht')
        polarity3 = self.model.polarity('wunderschön')
        polarity4 = self.model.polarity('mies')
        print(f'gut: {polarity1}, schlecht: {polarity2}, wunderschön: {polarity3}, mies: {polarity4}')
        assert polarity1 > 0
        assert polarity2 < 0
        assert polarity3 > 0
        assert polarity4 < 0


class TestTrainer(GenericTest):
    def _setup4(self):
        self._setup3()
        self.tr = trainer.TrainProcessor(lex=self.lex1,
                                         support_lex=self.lex2,
                                         embeddings=self.embeds)

    def _setup5(self):
        self._setup4()
        self.tr._prepare_inputs()

    def test_prep(self):
        self._setup2()
        lex1 = lexicon.Lexicon({'A1': ['Politik', 'Neuwahl', 'Koalition'], 'A2': ['Brot', 'Kuchen']})
        lex2 = lexicon.Lexicon({'B1': ['Lenkrad', 'tanken', 'Garage'], 'B2': ['ab', 'an']})
        cl = trainer.TrainProcessor(lex=lex1, embeddings=self.embeds)
        cl._handle_lexica(main_lex=lex1)
        assert isinstance(cl._main_lex, lexicon.Lexicon)
        assert cl._support_lex is None
        assert cl._main_keys == ['A1', 'A2']
        assert cl._support_keys == []

        cl._handle_lexica(main_lex=lex1, support_lex=lex2)
        assert isinstance(cl._main_lex, lexicon.Lexicon)
        assert isinstance(cl._support_lex, lexicon.Lexicon)
        assert cl._main_keys == ['A1', 'A2']
        assert cl._support_keys == ['B1', 'B2']

        cl._prepare_inputs()
        result = pd.DataFrame({'categories': ['A1', 'A1', 'A1',
                                              'A2', 'A2',
                                              'B1', 'B1', 'B1',
                                              'B2', 'B2'],
                                'terms': ['Politik', 'Neuwahl', 'Koalition',
                                          'Brot', 'Kuchen',
                                          'Lenkrad', 'tanken', 'Garage',
                                          'ab', 'an']})
        assert cl._term2cat.equals(result)
        assert cl._embedding_df.shape == (len(result), 300)
        assert list(cl._embedding_df.index.values)[:4] == ['A1:Politik',
                                                           'A1:Neuwahl',
                                                           'A1:Koalition',
                                                           'A2:Brot']

        y_classes =  cl._make_y()
        check1 = [True, True, True, False, False, False, False, False, False, False]
        check2 = [True, True, False, False, False, False, False, False, False, False]
        a1_ix = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        a2_ix = [3, 4, 0, 1, 2, 5, 6, 7, 8, 9]
        assert list(y_classes.keys()) == ['A1', 'A2']
        assert list(y_classes['A1']['A1']) == check1
        assert list(y_classes['A2']['A2']) == check2
        assert list(y_classes['A1'].index.values) == a1_ix
        assert list(y_classes['A2'].index.values) == a2_ix

        x, y = cl._make_data()
        cl._train_test_split(x, y, random_state=1)
        trains = cl.trains
        tests = cl.tests
        print(tests)
        assert list(trains.keys()) == ['A1', 'A2']
        assert list(tests.keys()) == ['A1', 'A2']
        assert len(trains['A1'])==2
        assert isinstance(trains['A1'][0], pd.DataFrame)
        assert isinstance(trains['A1'][1], pd.DataFrame)

    def test(self):
        self._setup5()
        self.tr.make_train_test_data()
        x, y = self.tr.feed_cat_Xy(cat = 'PolitikVR')
        print(x)
        print(y)
        print(y['categories'].unique())
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        assert x.shape[1] == 300
        assert y.shape[1] == 2
        assert list(y.columns) == ['PolitikVR', 'categories']
        uniques = sorted([str(x) for x in y['categories'].unique()])
        assert sorted([str(x) for x in uniques]) == sorted(['nan', 'AutoVR', 'Space', 'Food'])

    def test_embedding_df(self):
        self._setup4()
        df = self.tr._get_embedding_df(self.lex1)
        assert isinstance(df, pd.DataFrame)
        assert len(df.shape) == 2
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
        assert sorted(list(y.keys())) == sorted(list(self.lex1.keys))
        # print(y)
        # print(y['PolitikVR'])
        # assert False

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
        assert len(x['PolitikVR'].iloc[0, :]) == 300

    def test_make_lexicon(self):
        dct1 = {'A': ['a', 'b'], 'B': ['c', 'd', 'e']}
        dct2 = pd.DataFrame({'C': ['f', 'g'], 'D': ['h', 'i']})
        tr = trainer.TrainProcessor(lex=dct1)
        assert tr._main_keys == ['A', 'B']
        assert isinstance(tr._main_lex, lexicon.Lexicon)

        tr2 = trainer.TrainProcessor(lex=dct1, support_lex=dct2)
        assert tr2._main_keys == ['A', 'B']
        assert tr2._support_keys == ['C', 'D']
        assert isinstance(tr2._main_lex, lexicon.Lexicon)
        assert isinstance(tr2._support_lex, lexicon.Lexicon)

        tr3 = trainer.TrainProcessor(lex=dct1,
                                     main_keys=['A'],
                                     support_keys=['B'])
        assert tr3._main_keys == ['A']
        assert tr3._support_keys == ['B']


if __name__ == '__main__':
    unittest.main()
