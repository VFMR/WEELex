from typing import Union, Tuple, Iterable, List
import os
import json

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from weelex import lexicon
from weelex import embeddings
from weelex import base
from weelex.tfidf import BasicTfidf
from weelex.predictor import PredictionProcessor
from cluster_tfidf.cluster_tfidf.ctfidf import ClusterTfidfVectorizer
from batchprocessing import batchprocessing


class LatentSemanticScaling(base.BasePredictor):
    def __init__(self,
                 embeds: Union[dict, embeddings.Embeddings],
                #  polarity_lexicon: lexicon.Lexicon,
                 tfidf: Union[str, BasicTfidf] = None,
                 ctfidf: Union[str, ClusterTfidfVectorizer] = None,
                 use_ctfidf: bool = True,
                 random_state: int = None,
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
                 ) -> None:
        super().__init__(
            embeds=embeds,
            tfidf=tfidf,
            ctfidf=ctfidf,
            use_ctfidf=use_ctfidf,
            random_state=random_state,
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

    def polarity(self, word: str) -> float:
        if self._is_fit is False:
            raise NotFittedError('This LatentSemanticScaling instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.')
        return self._compute_polarity_word(word)

    def doc_polarity(self, doc: str) -> float:
        if self._predictprocessor is None:
            raise NotFittedError('This LatentSemanticScaling instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.')
        vector = self._predictprocessor.transform(pd.Series(np.array([doc])))
        return self._compute_polarity_vector(vector[0,:])

    def _compute_polarity_vector(self, vector: np.ndarray) -> float:
        lexicon_embeddings = self._lex.embeddings
        weights = self._lex.weights
        return self._polarity_function(vf=vector,
                                       Vs=lexicon_embeddings,
                                       P=weights)

    def _compute_polarity_word(self, word: str) -> float:
        vector = self._embeddings[word]
        return self._compute_polarity_vector(vector=vector)

    @staticmethod
    def _polarity_function(vf: np.ndarray,
                           Vs: np.ndarray,
                           P: np.ndarray) -> float:
        cosine_sum = 0
        for i in range(Vs.shape[0]):
            cosine_sum += cosine_simil(Vs[i,:], vf)*P[i]
        return (1 / Vs.shape[0]) * cosine_sum

    def fit(self,
            polarity_lexicon: lexicon.WeightedLexicon,
            X=Union[pd.Series, np.ndarray],
            y=None):
        self._lex = polarity_lexicon
        self._lex.embed(self._embeddings)

        # setting up the aggregator:
        self._fit_predictprocessor(X=X)

    @batchprocessing.batch_predict
    def predict_docs(self,
                     X: Union[pd.Series, np.ndarray],
                     n_batches: int = None,
                     checkpoint_path: str = None) -> np.ndarray:
        if self._predictprocessor is None:
            raise NotFittedError('This LatentSemanticScaling instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.')
        vects = self._predictprocessor.transform(X)
        return self.predict_vectors(vects)

    def predict_words(self,
                       X: Union[pd.Series, np.ndarray],
                       n_batches: int = None,
                       checkpoint_path: str = None) -> np.ndarray:
        vectors = np.zeros((X.shape[0], self._embeddings.dim))
        for i in range(X.shape[0]):
            vectors[i] = self._embeddings[X[i]]
        return self.predict_vectors(X=vectors)

    @batchprocessing.batch_predict
    def predict_vectors(self,
                        X: Union[pd.Series, np.ndarray],
                        n_batches: int = None,
                        checkpoint_path: str = None) -> np.ndarray:
        lexicon_embeddings = self._lex.embeddings
        weights = self._lex.weights
        scores = np.zeros((X.shape[0]))
        for i in range(X.shape[0]):
            scores[i] = self._polarity_function(vf=X[i,:],
                                               Vs=lexicon_embeddings,
                                               P=weights)
        return scores

def cosine_simil(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors

    Example:
        >>> a = np.array([1,2,3])
        >>> b = np.array([3,2,1])
        >>> cosine_simil(a, b)
        0.7142857142857143

    Args:
        a (np.ndarray): First vector
        b (np.ndarray): Second vector

    Returns:
        float: Cosine similarity scalar
    """
    # while technically undefined, we set the similarity to 0
    # when one of the vectors is a null vector.
    # Since we are mostly relying on FastText, this should normally not occur.
    if np.linalg.norm(a) != 0 and np.linalg.norm(b) != 0:
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    else:
        cos_sim = 0
    return cos_sim
