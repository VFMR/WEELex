from typing import Union, Iterable, Dict, List
import os

from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
# from tqdm import tqdm
import batchprocessing

from weelex import lexicon
from weelex import embeddings
from weelex import ensemble
from weelex import base
from weelex.trainer import TrainProcessor
from weelex.tfidf import BasicTfidf
# from weelex.predictor import PredictionProcessor
from cluster_tfidf.ctfidf import ClusterTfidfVectorizer


class WEELexClassifier(base.BasePredictor):
    def __init__(self,
                 embeds: Union[dict, embeddings.Embeddings],
                 tfidf: Union[str, BasicTfidf] = None,
                 ctfidf: Union[str, ClusterTfidfVectorizer] = None,
                 use_ctfidf: bool = True,
                 word_level_aggregation: bool = True,
                 test_size: float = None,
                 random_state: int = None,
                 n_jobs: int = 1,
                 progress_bar: bool = False,
                 relevant_pos: List[str] = ['ADJ', 'ADV', 'NOUN', 'VERB'],
                 min_df: Union[int, float] = 5,
                 max_df: Union[int, float] = 0.95,
                 spacy_model: str = 'de_core_news_lg',
                 n_docs: int = 2000000,
                 corpus_path: str = None,
                 corpus_path_encoding: str = 'latin1',
                 load_clustering: bool = False,
                 checkterm: str = 'Politik',
                 n_top_clusters: int = 3,
                 cluster_share: float = 0.2,
                 clustermethod: str = 'agglomerative',
                 distance_threshold: float = 0.5,
                 n_words: int = 40000,
                 **train_params) -> None:
        super().__init__(
            embeds=embeds,
            tfidf=tfidf,
            ctfidf=ctfidf,
            use_tfidf=True,
            use_ctfidf=use_ctfidf,
            word_level_aggregation=word_level_aggregation,
            random_state=random_state,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
            relevant_pos=relevant_pos,
            min_df=min_df,
            max_df=max_df,
            spacy_model=spacy_model,
            n_docs=n_docs,
            corpus_path=corpus_path,
            corpus_path_encoding=corpus_path_encoding,
            load_clustering=load_clustering,
            checkterm=checkterm,
            n_top_clusters=n_top_clusters,
            cluster_share=cluster_share,
            clustermethod=clustermethod,
            distance_threshold=distance_threshold,
            n_words=n_words
        )
        self._model = ensemble.FullEnsemble
        self._test_size = test_size
        self._train_params = train_params

        # setting up default objects
        self._support_lex = None
        self._main_keys = None
        self._support_keys = None
        self._models = {}
        self._best_params = {}
        self._tuned_params = {}
        self._cv_scores = {}
        self._results = {}

        if self._use_tfidf is False and self._use_ctfidf is False:
            raise ValueError('Both "use_tfidf" and "use_tfidf" are set to False. This is currently not implemented.')

    def set_params(self, **params):
        trainparams = {}
        for key, value in params.items():
            # CHECKME: check if this is a valid approach
            if key in self.__dict__:
                self.__dict__[key] == value
            else:
                trainparams.update({key: value})
        self._train_params = trainparams
        return self

    def fit(self,
            X: Union[pd.Series, np.ndarray],
            lex: Union[lexicon.Lexicon, dict, str],
            support_lex: Union[lexicon.Lexicon, dict, str] = None,
            main_keys: Iterable[str] = None,
            support_keys: Iterable[str] = None,
            hp_tuning: bool = False,
            n_iter: int = 150,
            cv: int = 5,
            param_grid: dict = None,
            fixed_params: dict = None,
            n_best_params: int = 3,
            progress_bar: bool = False) -> None:
        self._lex = lex
        self._support_lex = support_lex
        self._fit_predictprocessor(X=X)
        self._setup_trainprocessor(lex, support_lex, main_keys, support_keys)
        input_shape = self._trainprocessor.embedding_dim

        # loop over the categories:
        models = {}
        pb = self._set_progress_bar()
        for cat in pb(self._main_keys):
        # for cat in self._main_keys:
            if hp_tuning:
                self._hyperparameter_tuning(cat=cat,
                                            n_iter=n_iter,
                                            param_grid=param_grid,
                                            fixed_params=fixed_params,
                                            n_best_params=n_best_params,
                                            cv=cv)

            model_params = self._tuned_params.get(cat)
            if fixed_params is not None:
                fixed_params_dct = fixed_params
            else:
                fixed_params_dct = {}
            fixed_params_dct.update({'input_shape': input_shape})

            model = self._model(cat,
                                categories=self._trainprocessor.main_keys,
                                outside_categories=self._trainprocessor.support_keys,
                                param_set=model_params,
                                **fixed_params_dct)
            model.fit(*self._trainprocessor.feed_cat_Xy(cat=cat, train=True))
            models.update({cat: model})
        self._models = models
        self._is_fit = True

    def _get_full_vocab(self) -> list:
        all_terms = super()._get_full_vocab()
        all_terms += self._support_lex.vocabulary
        return list(set(all_terms))

    def save(self, path) -> None:
        super().save(path)
        if self._support_lex is not None:
            self._support_lex.save(os.path.join(path, 'support_lex'))

    def load(self, path) -> None:
        super().load(path)
        # TODO: load support lex

    def _setup_trainprocessor(self,
                              lex: Union[lexicon.Lexicon, dict, str],
                              support_lex: Union[lexicon.Lexicon, dict, str] = None,
                              main_keys: Iterable[str] = None,
                              support_keys: Iterable[str] = None,) -> None:
        self._trainprocessor = TrainProcessor(lex=lex,
                                              support_lex=support_lex,
                                              main_keys=main_keys,
                                              support_keys=support_keys,
                                              embeddings=self._embeddings,
                                              test_size=self._test_size,
                                              random_state=self._random_state)
        self._trainprocessor.make_train_test_data()
        self._main_keys = self._trainprocessor.main_keys
        self._support_keys = self._trainprocessor.support_keys

    def _hyperparameter_tuning(self,
                               cat,
                               n_iter,
                               param_grid,
                               fixed_params,
                               cv: int = 5,
                               n_best_params=3):
        if param_grid is None:
            raise ValueError('No parameter grid set for tuning.')

        if fixed_params is not None:
            fixed_params_dct = fixed_params
        else:
            fixed_params_dct = {}
        search = RandomizedSearchCV(
                estimator=ensemble.AugmentedEnsemble(cat,
                                                     categories=self._main_keys,
                                                     outside_categories=self._support_keys,
                                                     **fixed_params_dct),
                param_distributions=param_grid,
                n_iter=n_iter,
                n_jobs=self._n_jobs,
                cv=cv,
                scoring=None,
                random_state=self._random_state)

        X, y = self._trainprocessor.feed_cat_Xy(cat=cat, train=True)
        search.fit(X, y)
        self._best_params.update({cat: search.best_params_})
        results = search.cv_results_
        cv_scores = self._nonmissing_mean(results)
        cv_ranks = self._score_ranks(cv_scores)
        self._cv_scores.update({cat: cv_scores})
        self._results.update({cat: results})
        result_params = self._get_best_params(results, cv_ranks, n_best_params)
        self._tuned_params.update({cat: result_params})

    @staticmethod
    def _nonmissing_mean(results: dict) -> list:
        """ Returns the mean test scores from non-missing cross validation results.
        Only averages the results of splits with valid test scores.

        Examples:
            >>> wl = WEELexClassifier(embeds=None)
            >>> results = {'split0_test_score': np.array([np.nan, np.nan]), 'split1_test_score': np.array([0.6, 0.8]), 'split2_test_score': np.array([0.4, 0.8]), 'mean_test_score': np.array([np.nan, np.nan])}
            >>> wl._nonmissing_mean(results)
            [0.5, 0.8]

        Args:
            results (dict): cv_results_ from an
                sklearn.model_selection.RandomizedSearchCV instance

        Returns:
            list: mean test scores
        """
        test_scores = [list(value) for key, value in results.items() if 'test_score' in key and not 'mean' in key and not 'std' in key and not 'rank' in key]
        mean_test_scores = []
        for i in range(len(test_scores[0])):
            lst = np.array([x[i] for x in test_scores if not np.isnan(x[i])])
            mean_test_scores.append(np.mean(lst))
        return mean_test_scores

    @staticmethod
    def _score_ranks(mean_test_scores: list) -> list:
        """Rank the scores in descending order

        Examples:
            >>> wl = WEELexClassifier(embeds=None)
            >>> x = [0.1, 0.9, 0.5]
            >>> wl._score_ranks(x)
            [2, 0, 1]

        Args:
            mean_test_scores (list): list containing the mean test scores

        Returns:
            list: Rank with 0 being the highest score.
        """
        sortedlist = sorted(mean_test_scores, reverse=True)
        return [sortedlist.index(x) for x in mean_test_scores]

    @staticmethod
    def _get_best_params(results: dict, ranks: list, n_best_params: int) -> dict:
        """Select the n best parameter sets from a RandomizedSearchCV or
            GridSearchCV instance.

        Args:
            results (dict): truncated parameter set
        """
        result_params = []
        for x, y in zip(ranks,
                        results['params']):
            if x < n_best_params and len(result_params) < n_best_params:
                result_params.append(y)
        return result_params

    def _probas_to_binary(self, probas, cutoff):
        catpreds_binary = {}
        for cat in self._main_keys:
            if len(probas[cat].shape) == 2:
                pred = (probas[cat][:,0] >= cutoff).astype(int)
            else:
                pred = (probas[cat] >= cutoff).astype(int)
            catpreds_binary.update({cat: pred})
        return pd.DataFrame(catpreds_binary)

    @batchprocessing.batch_predict
    def predict_words(self,
                X: pd.DataFrame,
                cutoff: float = 0.5,
                n_batches: int = None,
                checkpoint_path: str = None) -> pd.DataFrame:
        catpreds = self.predict_proba_words(X=X)
        return self._probas_to_binary(catpreds, cutoff=cutoff)

    def predict_proba_words(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        catpreds = {}
        for cat in self._main_keys:
            model = self._models[cat]
            preds = model.predict_proba(X)
            catpreds.update({cat: preds})
        return catpreds

    @batchprocessing.batch_predict
    def predict_docs(self,
                       X: pd.DataFrame,
                       cutoff: float = 0.5,
                       n_batches: int = None,
                       checkpoint_path: str = None) -> pd.DataFrame:
        preds = self.predict_proba_docs(X=X)
        return self._probas_to_binary(preds, cutoff=cutoff)

    def predict_proba_docs(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._is_fit is False:
            raise NotFittedError(f'This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.')
        vects = self._predictprocessor.transform(X)
        return self.predict_proba_words(vects)

    def save(self, path):
        super().save(path)
        # TODO: Expand save() method beyond base class

    #---------------------------------------------------------------------------
    # properties
    @property
    def main_keys(self) -> list:
        return self._main_keys

    @property
    def support_keys(self) -> list:
        return self._support_keys

    @property
    def vocabulary(self):
        vocab = super().vocabulary
        if self._support_lex is not None:
            vocab += self._support_lex.vocabulary
        return list(set(vocab))

    #---------------------------------------------------------------------------
    # classmethods:
    @classmethod
    def load(cls, path):
        return super().load(path)
