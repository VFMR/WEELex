from typing import Union, Iterable, Dict, List

from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
# from tqdm import tqdm

from weelex import lexicon
from weelex import embeddings
from weelex import ensemble
from weelex import base
from weelex.trainer import TrainProcessor
from weelex.tfidf import BasicTfidf
# from weelex.predictor import PredictionProcessor
from cluster_tfidf.cluster_tfidf.ctfidf import ClusterTfidfVectorizer
from batchprocessing import batchprocessing


class WEELexClassifier(base.BasePredictor):
    def __init__(self,
                 embeds: Union[dict, embeddings.Embeddings],
                 tfidf: Union[str, BasicTfidf] = None,
                 ctfidf: Union[str, ClusterTfidfVectorizer] = None,
                 use_ctfidf: bool = True,
                 test_size: float = 0.2,
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
            use_ctfidf=use_ctfidf,
            test_size=test_size,
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
        # self._embeddings = self._make_embeddings(embeds)
        # self._is_fit = False
        self._model = ensemble.FullEnsemble
        # self._test_size = test_size
        # self._random_state = random_state
        self._n_jobs = n_jobs
        self._train_params = train_params
        # self._use_progress_bar = progress_bar
        # self._tfidf = tfidf
        # self._ctfidf = ctfidf
        # self._use_ctfidf = use_ctfidf
        # self._relevant_pos = relevant_pos
        # self._min_df = min_df
        # self._max_df = max_df
        # self._spacy_model = spacy_model
        # self._n_docs = n_docs
        # self._corpus_path = corpus_path
        # self._corpus_path_encoding = corpus_path_encoding
        # self._load_clustering = load_clustering
        # self._checkterm = checkterm
        # self._n_top_clusters = n_top_clusters
        # self._cluster_share = cluster_share
        # self._clustermethod = clustermethod
        # self._distance_threshold = distance_threshold
        # self._n_words = n_words

        # setting up default objects
        self._main_keys = None
        self._support_keys = None
        self._models = {}
        self._best_params = {}
        self._tuned_params = {}
        self._cv_scores = {}
        self._results = {}

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

    def get_params(self, deep: bool = True) -> dict:
        return self.__dict__

    def weelexfit(self,
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
    def predict(self,
                X: pd.DataFrame,
                cutoff: float = 0.5,
                n_batches: int = None,
                checkpoint_path: str = None) -> pd.DataFrame:
        catpreds = self.predict_proba(X=X)
        return self._probas_to_binary(catpreds, cutoff=cutoff)

    def predict_proba(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        catpreds = {}
        for cat in self._main_keys:
            model = self._models[cat]
            preds = model.predict_proba(X)
            catpreds.update({cat: preds})
        return catpreds

    def weelexpredict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        self._setup_predictprocessor()
        vects = self._predictprocessor.transform(X)
        return self.predict_proba(vects)

    @batchprocessing.batch_predict
    def weelexpredict(self,
                      X: pd.DataFrame,
                      cutoff: float = 0.5,
                      n_batches: int = None,
                      checkpoint_path: str = None) -> pd.DataFrame:
        preds = self.weelexpredict_proba(X=X)
        return self._probas_to_binary(preds, cutoff=cutoff)

    def fit_tfidf(self, data: Union[np.ndarray, pd.Series]) -> None:
        self._predictprocessor.fit_tfidf(data)

    def fit_ctfidf(self, data: Union[np.ndarray, pd.Series]) -> None:
        self._predictprocessor.fit_ctfidf(data)

    def save_tfidf(self, path: str) -> None:
        self._predictprocessor.save_tfidf(path)

    def save_ctfidf(self, dir: str) -> None:
        self._predictprocessor.save_ctfidf(dir)

    def load_tfidf(self, path: str) -> None:
        self._predictprocessor.load_tfidf(path)

    def load_ctfidf(self, path: str) -> None:
        self._predictprocessor.load_ctfidf(path)

    def save(self):
        # TODO: Implement save() method
        pass

    def load(self):
        # TODO: Implement load() method
        pass

    @property
    def main_keys(self) -> list:
        return self._main_keys

    @property
    def support_keys(self) -> list:
        return self._support_keys
