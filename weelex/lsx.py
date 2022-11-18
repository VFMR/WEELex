from typing import Union, Tuple, Iterable, List
from zipfile import ZipFile
import joblib

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from weelex import lexicon
from weelex import embeddings
from weelex import base
from weelex.tfidf import BasicTfidf
from cluster_tfidf.cluster_tfidf.ctfidf import ClusterTfidfVectorizer
from batchprocessing import batchprocessing


class LatentSemanticScaling(base.BasePredictor):
    def __init__(self,
                 embeds: Union[dict, embeddings.Embeddings],
                #  polarity_lexicon: lexicon.Lexicon,
                 word_level_aggregation: bool = False,
                 tfidf: Union[str, BasicTfidf] = None,
                 ctfidf: Union[str, ClusterTfidfVectorizer] = None,
                 use_tfidf: bool = True,
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
                 scale_results: bool = True,
                 ) -> None:
        super().__init__(
            embeds=embeds,
            tfidf=tfidf,
            ctfidf=ctfidf,
            use_ctfidf=use_ctfidf,
            word_level_aggregation=word_level_aggregation,
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
        self._use_result_scaling = scale_results
        self._use_tfidf =  use_tfidf
        self._scaler = StandardScaler()

    def _get_properties(self):
        properties =  super()._get_properties()
        properties.update({'use_result_scaling': self._use_result_scaling})
        return properties

    def _set_properties(self, properties):
        super()._set_properties(properties)
        self._use_result_scaling = properties['use_result_scaling']

    def _scale_results(self,
                       polarity_scores: Union[float, np.ndarray]
                       ) -> Union[float, np.ndarray]:
        if self._use_result_scaling:
            if isinstance(polarity_scores, float):
                score_array = np.array([polarity_scores]).reshape(-1,1)
                result = self._scaler.transform(score_array).reshape(-1)[0]
            else:
                score_array = polarity_scores.reshape(-1, 1)
                result = self._scaler.transform(score_array).reshape(-1)
        else:
            result = polarity_scores
        return result

    def polarity(self, word: str) -> float:
        if self._is_fit is False:
            raise NotFittedError('This LatentSemanticScaling instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.')
        return self._scale_results(self._compute_polarity_word(word))

    def doc_polarity(self, doc: str) -> float:
        if self._use_ctfidf:
            if self._predictprocessor is None:
                raise NotFittedError('This LatentSemanticScaling instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.')
            result = self._compute_polarity_ctfidf_doc(doc)
        elif self._use_tfidf:
            raise NotImplementedError('Prediction for tfidf but without ctfidf is not yet implemented.')
        else:
            lexicon_embeddings = self._lex.embeddings
            weights = self._lex.weights
            result = self._predict_doc_allwords(doc=doc,
                                                lexicon_embeddings=lexicon_embeddings,
                                                weights=weights)
        return self._scale_results(result)

    def _compute_polarity_ctfidf_doc(self, doc: str) -> float:
        vec_w = self._predictprocessor.transform(pd.Series(np.array([doc])))
        vectors = vec_w['vectors'][0]
        ctfidf_weights = vec_w['weights'][0]
        lexicon_embeddings = self._lex.embeddings
        polarity_weights = self._lex.weights
        polarity = self._compute_polarity_ctfidf_vecs(
            ctfidf_vectors=vectors,
            ctfidf_weights=ctfidf_weights,
            lexicon_embeddings=lexicon_embeddings,
            polarity_weights=polarity_weights
        )
        return polarity

    def _compute_polarity_ctfidf_vecs(self,
                                      ctfidf_vectors,
                                      ctfidf_weights,
                                      lexicon_embeddings,
                                      polarity_weights) -> float:
        polarity = 0
        for i, vf in enumerate(ctfidf_vectors):
            polarity += ctfidf_weights[i]*self._polarity_function(
                                            vf=vf,
                                            Vs=lexicon_embeddings,
                                            P=polarity_weights)
        return polarity

    def _compute_polarity_ctfidf_corpus(self, corpus: pd.Series) -> np.ndarray:
        result = np.zeros( (len(corpus)) )
        vec_w = self._predictprocessor.transform(corpus)
        lexicon_embeddings = self._lex.embeddings
        polarity_weights = self._lex.weights

        for i in range(len(corpus)):
            ctfidf_vectors = vec_w['vectors'][i]
            ctfidf_weights = vec_w['weights'][i]
            result[i] = self._compute_polarity_ctfidf_vecs(
                ctfidf_vectors=ctfidf_vectors,
                ctfidf_weights=ctfidf_weights,
                lexicon_embeddings=lexicon_embeddings,
                polarity_weights=polarity_weights
                )
        return result

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
            cosine_sum += _cosine_simil(Vs[i,:], vf)*P[i]
        return (1 / Vs.shape[0]) * cosine_sum

    def fit(self,
            polarity_lexicon: lexicon.WeightedLexicon,
            X=Union[pd.Series, np.ndarray],
            y=None):
        self._lex = polarity_lexicon
        self._lex.embed(self._embeddings)

        # setting up the aggregator:
        self._fit_predictprocessor(X=X)
        self._is_fit = True

        # making prediction on X to fit scaler:
        preds = self._predict_docs_unscaled(X)
        self._scaler.fit(preds.reshape(-1,1))

    def _transform_doc_unweighted(self, doc: str) -> np.ndarray:
        splits = doc.split()
        result = []
        for i, x in enumerate(splits):
            if x in self._embeddings.keys:
                result.append(self._embeddings[x])
        return result

    def _predict_doc_allwords(self, doc, lexicon_embeddings, weights) -> float:
        row_vects = self._transform_doc_unweighted(doc)
        row_scores = []
        for x in row_vects:
            row_scores.append(
                self._polarity_function(vf=x,
                                        Vs=lexicon_embeddings,
                                        P=weights))
        return np.mean(row_scores)

    def _predict_docs_unscaled(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        if self._predictprocessor is None:
            raise NotFittedError('This LatentSemanticScaling instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.')

        if self._use_ctfidf or self._use_tfidf:
            preds = self._compute_polarity_ctfidf_corpus(X)
        else:
            lexicon_embeddings = self._lex.embeddings
            weights = self._lex.weights
            preds = np.zeros( (X.shape[0]))
            for i in range(X.shape[0]):
                doc = X[i]
                preds[i] = self._predict_doc_allwords(
                    doc=doc,
                    lexicon_embeddings=lexicon_embeddings,
                    weights=weights)
        return preds


    @batchprocessing.batch_predict
    def predict_docs(self,
                     X: Union[pd.Series, np.ndarray],
                     n_batches: int = None,
                     checkpoint_path: str = None) -> np.ndarray:
        preds = self._predict_docs_unscaled(X=X)
        return self._scale_results(preds)

    @batchprocessing.batch_predict
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
        if self._word_level_aggregation:
            for i in range(X.shape[0]):
                scores[i] = self._polarity_function(vf=X[i,:],
                                                Vs=lexicon_embeddings,
                                                P=weights)
        else:
            for i in range(X.shape[0]):
                row_vectors = X['vectors'][i]
                row_weights = X['weights'][i]
                row_score = 0
                for j, x in enumerate(row_vectors):
                    row_score += row_weights[j] * self._polarity_function(vf=x,
                                                         Vs=lexicon_embeddings,
                                                         P=weights)
                scores[i] = row_score

        return self._scale_results(scores)

    #---------------------------------------------------------------------------
    # classmethods:
    @classmethod
    def load(cls, path: str) -> 'LatentSemanticScaling':
        instance = super().load(path)
        usepath = cls._check_zippath(path)
        with ZipFile(usepath) as myzip:
            with myzip.open('scaler.joblib', 'r') as f:
                instance._scaler = joblib.load(f)
        return instance


def _cosine_simil(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors

    Example:
        >>> a = np.array([1,2,3])
        >>> b = np.array([3,2,1])
        >>> _cosine_simil(a, b)
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
