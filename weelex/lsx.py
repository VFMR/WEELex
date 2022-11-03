from typing import Union, Tuple, Iterable, List

import numpy as np
import pandas as pd

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
                 polarity_lexicon: lexicon.Lexicon,
                 tfidf: Union[str, BasicTfidf] = None,
                 ctfidf: Union[str, ClusterTfidfVectorizer] = None,
                 use_ctfidf: bool = True,
                 test_size: float = 0.2,
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
            test_size=test_size,
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
        self._polarity_lexicon = polarity_lexicon

    def _compute_polarity(self, word: str) -> float:
        all_similarities = []
        for t in self._polarity_lexicon.keys:

            all_similarities.append()

    def fit(self, X, y):
        pass

    @batchprocessing.batch_predict
    def predict_docs(self,
                X: pd.DataFrame,
                cutoff: float = 0.5,
                n_batches: int = None,
                checkpoint_path: str = None) -> np.ndarray:
        preds = self.predict_score_docs(X=X)
        return preds

    def predict_score_docs(self, X: pd.DataFrame) -> np.ndarray:
        self._setup_predictprocessor()
        vects = self._predictprocessor.transform(X)
        return self.predict_score_words(vects)

    @batchprocessing.batch_predict
    def predict_words(self,
                      X: pd.DataFrame,
                      cutoff: float = 0.5,
                      n_batches: int = None,
                      checkpoint_path: str = None) -> np.ndarray:
        preds = self.predict_score_words(X=X)
        return preds

    def predict_score_words(self, X: pd.DataFrame) -> np.ndarray:
        # TODO: implement function to actually make the predictions
        pass




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
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
