from json import load
from re import I
from typing import Union, List

import numpy as np
import pandas as pd

from weelex.cluster_tfidf import ctfidf
from weelex.tfidf import BasicTfidf
from weelex import embeddings

class PredictionProcessor:
    def __init__(self,
                 data: Union[np.ndarray, pd.Series] = None,
                 embeddings: embeddings.Embeddings = None,
                 tfidf: BasicTfidf = None,
                 ctfidf: ctfidf.ClusterTfidfVectorizer = None,
                 relevant_pos: List[str] = ['ADJ', 'ADV', 'NOUN', 'VERB'],
                 min_df: Union[int, float] = 5,
                 max_df: Union[int, float] = 0.95,
                 spacy_model: str = 'de_core_news_lg',
                 n_docs: int = 2000000,
                 corpus_path: str = None,
                 corpus_path_encoding: str = 'latin1',
                 load_clustering: bool = False,
                 embedding_dim: int = False,
                 checkterm: str = 'Politik',
                 n_top_clusters: int = 3,
                 cluster_share: float = 0.2,
                 clustermethod: str = 'agglomerative',
                 distance_threshold: float = 0.5,
                 n_words: int = 40000,
                 n_jobs: int = 1) -> None:
        self._data = data
        self._embeddings = embeddings
        self._relevant_pos = relevant_pos
        self._min_df = min_df
        self._max_df = max_df
        self._spacy_model = spacy_model
        self._n_docs = n_docs
        self._corpus_path = corpus_path
        self._corpus_path_encoding = corpus_path_encoding
        self._load_clustering = load_clustering
        self._embedding_dim = embedding_dim
        self._checkterm = checkterm
        self._n_top_clusters = n_top_clusters
        self._cluster_share = cluster_share
        self._clustermethod = clustermethod
        self._distance_threshold = distance_threshold
        self._n_words = n_words
        self._n_jobs = n_jobs

        if tfidf is not None:
            self._tfidf = tfidf
        else:
            self._tfidf = None

        if ctfidf is not None:
            self._ctfidf = ctfidf
        else:
            self._ctfidf = None

    def _instantiate_tfidf(self):
        tfidf = BasicTfidf(relevant_pos=self._relevant_pos,
                            min_df=self._min_df,
                            max_df=self._max_df,
                            spacy_model=self._spacy_model)
        return tfidf

    def _instantiate_ctfidf(self) -> ctfidf.ClusterTfidfVectorizer:
        vectorizer = ctfidf.ClusterTfidfVectorizer(
                    vectorizer=self._tfidf,
                    embeddings=self._embeddings,
                    n_docs=self._n_docs,
                    corpus_path=self._corpus_path,
                    corpus_path_encoding=self._corpus_path_encoding,
                    load_clustering=self._load_clustering,
                    embedding_dim=self._embedding_dim,
                    checkterm=self._checkterm,
                    n_top_clusters=self._n_top_clusters,
                    cluster_share=self._cluster_share,
                    clustermethod=self._clustermethod,
                    distance_threshold=self._distance_threshold,
                    n_words=self._n_words,
                    n_jobs=self._n_jobs)
        return vectorizer

    def fit_tfidf(self, data: Union[np.ndarray, pd.Series] = None) -> None:
        if data is None:
            usedata = self._data
        else:
            usedata = data

        if self._tfidf is None:
            self._tfidf = self._instantiate_tfidf()

        self._tfidf.fit(usedata)

    def process(self):
        pass
