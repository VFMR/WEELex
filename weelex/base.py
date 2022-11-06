from typing import Union, List
import os
import json

from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError

from weelex import lexicon
from weelex import embeddings
from weelex.tfidf import BasicTfidf
from weelex.predictor import PredictionProcessor
from cluster_tfidf.cluster_tfidf.ctfidf import ClusterTfidfVectorizer
from batchprocessing import batchprocessing


class BasePredictor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 embeds: Union[dict, embeddings.Embeddings],
                 tfidf: Union[str, BasicTfidf] = None,
                 ctfidf: Union[str, ClusterTfidfVectorizer] = None,
                 use_ctfidf: bool = True,
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
                 n_words: int = 40000) -> None:
        self._embeddings = self._make_embeddings(embeds)
        # self._model = ensemble.FullEnsemble
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._use_progress_bar = progress_bar
        self._tfidf = tfidf
        self._ctfidf = ctfidf
        self._use_ctfidf = use_ctfidf
        self._relevant_pos = relevant_pos
        self._min_df = min_df
        self._max_df = max_df
        self._spacy_model = spacy_model
        self._n_docs = n_docs
        self._corpus_path = corpus_path
        self._corpus_path_encoding = corpus_path_encoding
        self._load_clustering = load_clustering
        self._checkterm = checkterm
        self._n_top_clusters = n_top_clusters
        self._cluster_share = cluster_share
        self._clustermethod = clustermethod
        self._distance_threshold = distance_threshold
        self._n_words = n_words

        # initialize properties:
        self._predictprocessor = None
        self._is_fit = False
        self._lex = None

    def get_params(self, deep: bool = True) -> dict:
        return self.__dict__

    def _setup_predictprocessor(self):
        self._predictprocessor = PredictionProcessor(
            embeddings=self._embeddings,
            tfidf=self._tfidf,
            ctfidf=self._ctfidf,
            use_ctfidf=self._use_ctfidf,
            relevant_pos=self._relevant_pos,
            min_df=self._min_df,
            max_df=self._max_df,
            spacy_model=self._spacy_model,
            n_docs=self._n_docs,
            corpus_path=self._corpus_path,
            corpus_path_encoding=self._corpus_path_encoding,
            load_clustering=self._load_clustering,
            checkterm=self._checkterm,
            n_top_clusters=self._n_top_clusters,
            cluster_share=self._cluster_share,
            clustermethod=self._clustermethod,
            distance_threshold=self._distance_threshold,
            n_words=self._n_words,
            n_jobs=self._n_jobs)

    def _load_predictprocessor(self, path):
        self._predictprocessor = PredictionProcessor(
            embeddings=self._embeddings
        )
        self._predictprocessor.load(os.path.join(path, 'predictprocessor'))
        # HACK: setting the private _embeddings property is not optimal
        self._predictprocessor._embeddings = self._embeddings

    def fit_tfidf(self, data: Union[np.ndarray, pd.Series]) -> None:
        self._predictprocessor.fit_tfidf(data)

    def fit_ctfidf(self, data: Union[np.ndarray, pd.Series]) -> None:
        self._predictprocessor.fit_ctfidf(data)

    # def save_tfidf(self, path: str) -> None:
    #     self._predictprocessor.save_tfidf(path)

    # def save_ctfidf(self, dir: str) -> None:
    #     self._predictprocessor.save_ctfidf(dir)

    # def load_tfidf(self, path: str) -> None:
    #     self._predictprocessor.load_tfidf(path)

    # def load_ctfidf(self, path: str) -> None:
    #     self._predictprocessor.load_ctfidf(path)

    def _fit_predictprocessor(self, X: Union[pd.Series, np.ndarray]) -> None:
        if self._predictprocessor is None:
            self._setup_predictprocessor()
        self._predictprocessor.fit(X=X)
        self._is_fit = True

    def save(self, path):
        if self._is_fit is False:
            raise NotFittedError(f'This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.')

        os.makedirs(path, exist_ok=True)
        self._predictprocessor.save(os.path.join(path, 'predictprocessor'))
        properties = self._get_properties()
        with open(os.path.join(path, 'properties.json'), 'w') as f:
            json.dump(properties, f)

        # filter embeddings if not done already and save the relevant vectors
        if self._embeddings.isfiltered is False:
            self._embeddings.filter_terms(self._get_full_vocab())
        self._embeddings.save_filtered(os.path.join(path, 'embeddings'))

        if self._lex is not None:
            self._lex.save(os.path.join(path, 'lex'))

    def _get_full_vocab(self):
        all_words = []
        if self._lex is not None:
            all_words += self._lex.vocabulary
        all_words += self._tfidf.vocabulary_
        return list(set(all_words))


    def load(self, path):
        with open(os.path.join(path, 'properties.json'), 'r') as f:
            properties = json.load(f)

        self._set_properties(properties=properties)
        self._embeddings.load_filtered(os.path.join(path, 'embeddings'))
        self._load_predictprocessor(path)
        # TODO: load lex

    def _get_properties(self):
        properties = {
        'use_ctfidf': self._use_ctfidf,
        'random_state': self._random_state,
        'progress_bar': self._use_progress_bar,
        'relevant_pos': self._relevant_pos,
        'min_df': self._min_df,
        'max_df': self._max_df,
        'spacy_model': self._spacy_model,
        'n_docs': self._n_docs,
        'corpus_path': self._corpus_path,
        'corpus_path_encoding': self._corpus_path_encoding,
        'load_clustering': self._load_clustering,
        'checkterm': self._checkterm,
        'n_top_clusters': self._n_top_clusters,
        'cluster_share': self._cluster_share,
        'clustermethod': self._clustermethod,
        'distance_threshold': self._distance_threshold,
        'n_words': self._n_words}
        return properties

    def _set_properties(self, properties):
        self._use_ctfidf = properties['use_ctfidf']
        self._random_state = properties['random_state']
        self._use_progress_bar = properties['progress_bar']
        self._relevant_pos = properties['relevant_pos']
        self._min_df = properties['min_df']
        self._max_df = properties['max_df']
        self._spacy_model = properties['spacy_model']
        self._n_docs = properties['n_docs']
        self._corpus_path = properties['corpus_path']
        self._corpus_path_encoding = properties['corpus_path_encoding']
        self._load_clustering = properties['load_clustering']
        self._checkterm = properties['checkterm']
        self._n_top_clusters = properties['n_top_clusters']
        self._cluster_share = properties['cluster_share']
        self._clustermethod = properties['clustermethod']
        self._distance_threshold = properties['distance_threshold']
        self._n_words = properties['n_words']

    def _set_progress_bar(self):
        if self._use_progress_bar:
            return tqdm
        else:
            return self._emptyfunc

    @staticmethod
    def _emptyfunc(array):
        return array

    @staticmethod
    def _make_embeddings(embeds: Union[embeddings.Embeddings, dict]):
        if not isinstance(embeds, embeddings.Embeddings):
            my_embeds = embeddings.Embeddings(embedding_dict=embeds)
        else:
            my_embeds = embeds
        return my_embeds

    def _get_full_vocab(self) -> list:
        v1 = self._lexicon.get_vocabulary()
        v2 = self._support_lexicon.get_vocabulary()
        return sorted(list(set([v1, v2])))

    def _filter_embeddings(self) -> None:
        vocab = self._get_full_vocab()
        self._embeddings.filter_terms(vocab)

    def __repr__(self, N_CHAR_MAX=700):
        return super().__repr__(N_CHAR_MAX)

