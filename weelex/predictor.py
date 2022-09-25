from json import load
from re import I
from typing import Union, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from weelex.tfidf import BasicTfidf
from weelex import embeddings
from cluster_tfidf.cluster_tfidf.ctfidf import ClusterTfidfVectorizer

class PredictionProcessor:
    def __init__(self,
                 data: Union[np.ndarray, pd.Series] = None,
                 embeddings: embeddings.Embeddings = None,
                 tfidf: Union[str, BasicTfidf] = None,
                 ctfidf: Union[str, ClusterTfidfVectorizer] = None,
                 use_ctfidf: bool = True,
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
                 n_jobs: int = 1) -> None:
        self._data = data
        self._embeddings = embeddings
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
        self._n_jobs = n_jobs

        self._embedding_dim = self._get_embedding_dim(self._embeddings,
                                                      self._checkterm)

        if tfidf is not None:
            if isinstance(tfidf, str):
                self.load_tfidf(tfidf)
            elif isinstance(tfidf, BasicTfidf):
                self._tfidf = tfidf
            else:
                raise ValueError(f"""
                    Expected 'tfidf' to be of type 'str' or 'BasicTfidf'.
                    Got {type(tfidf)} instead.
                """)
        else:
            self._tfidf = None

        if ctfidf is not None:
            if isinstance(ctfidf, str):
                self.load_ctfidf(ctfidf)
            elif isinstance(ctfidf, ClusterTfidfVectorizer):
                self._ctfidf = ctfidf
            else:
                raise ValueError(f"""
                    Expected 'ctfidf' to be of type 'str' or 'ClusterTfidfVectorizer'.
                    Got {type(ctfidf)} instead.
                """)
        else:
            self._ctfidf = None

    def _instantiate_tfidf(self):
        tfidf = BasicTfidf(relevant_pos=self._relevant_pos,
                            min_df=self._min_df,
                            max_df=self._max_df,
                            spacy_model=self._spacy_model)
        return tfidf

    def _instantiate_ctfidf(self) -> ClusterTfidfVectorizer:
        if isinstance(self._tfidf, BasicTfidf):
            tfidf = self._tfidf.vectorizer
        vectorizer = ClusterTfidfVectorizer(
                    vectorizer=tfidf,
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

    @staticmethod
    def _checksize(array: np.ndarray) -> None:
        if len(array.shape) != 1:
            raise ValueError(f"""
                Expected 1 dimensional array. Got array of shape {array.shape}.
            """)

    def _get_usedata(self, data):
        if data is None:
            usedata = self._data
        else:
            self._checksize(data)
            usedata = data
        return usedata

    def fit_tfidf(self, data: Union[np.ndarray, pd.Series] = None) -> None:
        usedata = self._get_usedata(data)
        if self._tfidf is None:
            self._tfidf = self._instantiate_tfidf()

        self._tfidf.fit(usedata)

    def _vectorize(self, data):
        if self._use_ctfidf:
            return self._ctfidf.transform(data)
        else:
            return self._tfidf.transform(data)

    def fit_ctfidf(self, data: Union[np.ndarray, pd.Series] = None) -> None:
        usedata = self._get_usedata(data)
        if self._ctfidf is None:
            self._ctfidf = self._instantiate_ctfidf()
        self._ctfidf.fit()

    def save_tfidf(self, path: str) -> None:
        self._tfidf.save(path)

    def load_tfidf(self, path: str) -> None:
        tfidf = BasicTfidf()
        tfidf.load(path)
        self._tfidf = tfidf

    def save_ctfidf(self, dir: str) -> None:
        self._ctfidf.save(dir)

    def load_ctfidf(self, path: str) -> None:
        ctfidf = ClusterTfidfVectorizer()
        ctfidf.load(path)
        self._ctfidf = ctfidf

    def transform(self, X: Union[np.ndarray, pd.Series] = None):
        usedata = self._get_usedata(X)

        if self._tfidf is None or not self._tfidf.check_fit():
            self.fit_tfidf(usedata)

        # ctfidf has aggregation of words already built in
        if self._use_ctfidf:
            if self._ctfidf is None:
                self.fit_ctfidf(usedata)
            result = self._vectorize(usedata)
        else:
            vects = self._vectorize(usedata)
            # additional step of aggregation necessary if ctfidf is not used
            result = self._tfidf_weighted_embeddings_corpus(vects)

        return result

    @staticmethod
    def _get_word_tfidf_pairs(vect, index2word):
        """Reorganize the sparse tfidf vector, i.e. one row
        of the sparse tfidf matrix, into a vector containing
        only the nonzero elements of the vector.

        Args:
            vect (row of sparse matrix): One row of the sparse matrix resulting from
                tfidf vectorization.
            index2word (dict): Lookup table word index-> word string

        Returns:
            list: Array containing the terms (str) with nonzero vectors
            numpy.array: vector containing the nonzero tfidf values
        """
        if not isinstance(vect, np.ndarray):
            vect_array = vect.toarray()[0]
        else:
            vect_array = vect
        indices = list(np.where(vect_array != 0)[0])
        words = [index2word[x] for x in indices]
        nonzero_vects = np.array(vect_array[indices])
        return words, nonzero_vects

    def _tfidf_weighted_embeddings(self, vect, index2word):
        """Aggregate embedding vectors of rows as a linear combination with
        the respective words' Tfidf values as weights (normalized to sum to 1)

        Args:
            vect (sparse vector): One row of the sparse matrix resulting from tfidf
                vectorization.
            index2word (dict): Lookup table word index -> word string
            wv (dict or gensim.models.Fasttext): Lookup for embedding vectors
            top_n (int, optional): If int: number of terms to aggregate. Allows
                to only aggregate the n terms with the highest tfidf values. Defaults to None.

        Returns:
            numpy.array: vector containing the aggregated word embedding
        """
        words, weights = self._get_word_tfidf_pairs(vect, index2word)
        if self._n_top_clusters:
            n = min(self._n_top_clusters, len(weights))
            words = [x for _, x in sorted(zip(weights, words))]
            weights = [y for y, _ in sorted(zip(weights, words))]
            words = words[-n:]
            weights = np.array(weights[-n:])
        weights = weights / weights.sum()
        ft_array = np.array([self._embeddings[word] for word in words]).T
        weighted_sum = ft_array @ weights
        return weighted_sum

    def _tfidf_weighted_embeddings_corpus(self, vects):
        """Aggregate the embedding vectors of all rows in the data

        Args:
            vects (sparse matrix): Result from tfidf vectorization of dataset
            wv (dict or gensim.models.Fasttext): Lookup for embedding vectors of words
            tfidf (sklearn.feature_processing.text.TfidfVectorizer instance): The Tfidf Vectorizer instance
                that has been fitted before.
            top_n (int, optional): If int: number of terms to aggregate. Allows
                to only aggregate the n terms with the highest tfidf values. Defaults to None.

        Returns:
            numpy.array: Matrix of size (len(data) x embedding dimensions)
        """
        embedding_shape = self._embedding_dim
        index2word = {i: term for term, i in self._tfidf.vocabulary_.items()}

        print('{} rows to process'.format(vects.shape[0]))
        allvects = np.zeros((vects.shape[0], embedding_shape))
        i = 0
        for vect in tqdm(vects):
            allvects[i] = self._tfidf_weighted_embeddings(vect, index2word)
            i += 1

        return allvects

    @staticmethod
    def _get_embedding_dim(embeddings, checkterm):
        return len(embeddings[checkterm])

    @property
    def embedding_dim(self):
        return self._get_embeddings_dim(self._embeddings, self._checkterm)
