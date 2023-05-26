"""
Contains the base prediction class which is inherited by
the both the weelex and lsx classifiers.
"""
from typing import Union, List
import os
import json
import shutil
from zipfile import ZipFile
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
import spacy
from cluster_tfidf.ctfidf import ClusterTfidfVectorizer

from weelex import lexicon
from weelex import embeddings
from weelex.tfidf import BasicTfidf
from weelex._predictor import _PredictionProcessor


class _BasePredictor(BaseEstimator, TransformerMixin):
    """
    Base prediction class with methods for both the weelex and lsx
    classifiers.
    """

    def __init__(
        self,
        embeds: Union[dict, embeddings.Embeddings],
        tfidf: Union[str, BasicTfidf] = None,
        ctfidf: Union[str, ClusterTfidfVectorizer] = None,
        use_tfidf: bool = True,
        use_ctfidf: bool = True,
        word_level_aggregation: bool = True,
        random_state: int = None,
        n_jobs: int = 1,
        progress_bar: bool = False,
        relevant_pos: List[str] = ["ADJ", "ADV", "NOUN", "VERB"],
        min_df: Union[int, float] = 5,
        max_df: Union[int, float] = 0.95,
        spacy_model: str = "de_core_news_lg",
        n_docs: int = 2000000,
        corpus_path: str = None,  # TODO: Check if really needed in ctfidf
        corpus_path_encoding: str = "latin1",  # TODO: Check if needed in ctfidf
        load_clustering: bool = False,  # CHECKME: Check what this is doing
        checkterm: str = "Politik",
        n_top_clusters: int = 3,
        cluster_share: float = 0.2,
        clustermethod: str = "agglomerative",
        distance_threshold: float = 0.5,
        n_words: int = 40000,
    ) -> None:
        """Initialization for the class.

        Args:
            embeds (Union[dict, embeddings.Embeddings]): Word to embedding
                vectors lookup.
            tfidf (Union[str, BasicTfidf], optional): Tfidf implementation.
                Can either be a fitted weelex.tfidf.BasicTfidf instance
                or the path to a saved instance.
                If None, a new weelex.tfidf.BasicTfidf instance is instantiated
                and fitted.
                Defaults to None.
            ctfidf (Union[str, ClusterTfidfVectorizer], optional): Cluster Tfidf
                implementation. Can either be a fitted
                cluster_tfidf.ctfidf.ClusterTfidfVectorizer instance or a path
                to a saved instance. If None, a new
                cluster_tfidf.ctfidf.ClusterTfidfVectorizer instance is
                instantiated and fitted. Defaults to None.
            use_tfidf (bool, optional): Whehter to use tfidf or not.
                Defaults to True.
            use_ctfidf (bool, optional): Whether to use Cluster Tfidf or not.
                Defaults to True.
            word_level_aggregation (bool, optional): Whether the output
                of document level inputs shall be aggregated on a word level.
                Defaults to True.
            random_state (int, optional): Random seed for replicability.
                Defaults to None.
            n_jobs (int, optional): Number of parallel processes to use.
                insert -1 to use all available cores.
                Defaults to 1.
            progress_bar (bool, optional): Whether to show a progress bar.
                Defaults to False.
            relevant_pos (List[str], optional): Only words with these Part of
                Speech (PoS) tags will be utilized. Possible values are
                "ADJ" for adjectives, "ADV" for adverbs, "NOUN" for nouns, and
                "VERB" for verbs. Defaults to ["ADJ", "ADV", "NOUN", "VERB"].
            min_df (Union[int, float], optional): Words need to be in at least
                this many documents to be considered. Defaults to 5.
            max_df (Union[int, float], optional): Words must be in fewer than
                these documents to be considered. `int` for the number of
                documents and `float` for the share of documents.
                Defaults to 0.95.
            spacy_model (str, optional): Name of the spacy model used for
                POS-tagging.
                See the (spaCy documentation)[https://spacy.io/usage/models] for
                info on available models. Defaults to "de_core_news_lg".
            n_docs (int, optional): Number of documents to use for fitting
                the tfidf and cluster-tfidf instances. Defaults to 2000000.
            corpus_path (str, optional): Path to the training corpus.
                Defaults to None.
            corpus_path_encoding (str, optional): Encoding of training corpus.
                Defaults to "latin1".
            load_clustering (bool, optional): Whether or not the clusters
                shall be loaded. Defaults to False.
            checkterm (str, optional): Word to validate the embeddings against.
                Needs to be included among the embedding vectors.
                Defaults to "Politik".
            n_top_clusters (int, optional): Number of top clusters or words to
                be aggregated. Defaults to 3.
            cluster_share (float, optional): Defaults to 0.2.
            clustermethod (str, optional): Method for cluster tfidf word
                clustering. Currently, only {"agglomerative"} are supported.
                Defaults to "agglomerative".
            distance_threshold (float, optional): Distance threshold parameter
                for cluster-tfidf clustering. Defaults to 0.5.
            n_words (int, optional): Number of words to cluster. More words
                come with larger memory requirements. Defaults to 40000.

        Raises:
            ValueError: If both use_tfidf and use_ctfidf are set to False
        """
        self._embeddings = self._make_embeddings(embeds, checkterm=checkterm)
        # self._model = ensemble.FullEnsemble
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._use_progress_bar = progress_bar
        self._tfidf = tfidf
        self._ctfidf = ctfidf
        self._use_tfidf = use_tfidf
        self._use_ctfidf = use_ctfidf
        self._word_level_aggregation = word_level_aggregation
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

        if self._use_tfidf is False and self._use_ctfidf is False:
            raise ValueError('Both "use_tfidf" and "use_ctfidf" are set to False.')

        # initialize properties:
        self._predictprocessor = None
        self._is_fit = False
        self._lex = None
        self._scaler = None

        # load spacy or download alternatively:
        try:
            spacy.load(self._spacy_model)
        except OSError as e:
            print(e)
            print(f"Downloading spacy model {self._spacy_model}.")
            spacy.cli.download(self._spacy_model)
            spacy.load(self._spacy_model)

    def get_params(self, deep: bool = False) -> dict:
        """Retrieve the parameters for the instance.

        Returns:
            dict: Set of parameters.
        """
        return self.__dict__

    def _setup_predictprocessor(self):
        self._predictprocessor = _PredictionProcessor(
            embeddings=self._embeddings,
            tfidf=self._tfidf,
            ctfidf=self._ctfidf,
            use_tfidf=self._use_tfidf,
            use_ctfidf=self._use_ctfidf,
            aggregate_word_level=self._word_level_aggregation,
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
            n_jobs=self._n_jobs,
        )

    def _load_predictprocessor(self, zip_archive):
        # self._predictprocessor = _PredictionProcessor(
        #     embeddings=self._embeddings
        # )
        # self._predictprocessor.load(os.path.join(path, 'predictprocessor'))
        # # HACK: setting the private _embeddings property is not optimal
        # self._predictprocessor._embeddings = self._embeddings
        self._predictprocessor = _PredictionProcessor.load_from_weelexarchive(
            zip_archive
        )

    def fit_tfidf(self, data: Union[np.ndarray, pd.Series]) -> None:
        """Method to fit the tfidf instance given a training corpus.

        Args:
            data (Union[np.ndarray, pd.Series]): Corpus with training documents.
        """
        self._predictprocessor.fit_tfidf(data)

    def fit_ctfidf(self, data: Union[np.ndarray, pd.Series]) -> None:
        """Method to fit the cluster-tfidf instance given a training corpus.

        Args:
            data (Union[np.ndarray, pd.Series]): Corpus with training documents.
        """
        self._predictprocessor.fit_ctfidf(data)

    def _fit_predictprocessor(self, X: Union[pd.Series, np.ndarray]) -> None:
        if self._predictprocessor is None:
            self._setup_predictprocessor()
        self._predictprocessor.fit(X=X, keepwords=self._lex.vocabulary)

    def save(self, path: str) -> None:
        """Save the trained model to disk into a compressed archive.

        Args:
            path (str): path to write the model to.
        """
        self._save_objects_to_dir(path)
        self._clean_save(path)

    def _save_objects_to_dir(self, path):
        # TODO: directly write into zip archive instead of rmtree()
        if self._is_fit is False:
            raise NotFittedError(
                f"""This {self.__class__.__name__} instance is not fitted yet.
                Call "fit" with appropriate arguments before using_
                this estimator."""
            )

        os.makedirs(path, exist_ok=True)
        self._predictprocessor.save(os.path.join(path, "predictprocessor"))
        properties = self._get_properties()
        with open(os.path.join(path, "properties.json"), "w") as f:
            json.dump(properties, f)

        # filter embeddings if not done already and save the relevant vectors
        if self._embeddings.isfiltered is False:
            self._embeddings.filter_terms(self._get_full_vocab())
        self._embeddings.save_filtered(os.path.join(path, "embeddings"))

        if self._lex is not None:
            self._lex.save(os.path.join(path, "lex"))

        if self._scaler is not None:
            joblib.dump(self._scaler, os.path.join(path, "scaler.joblib"))

    def _clean_save(self, path):
        # archiving the created folder
        shutil.make_archive(path + ".weelex", "zip", path)
        shutil.rmtree(path)

    def _get_full_vocab(self):
        all_words = []
        if self._lex is not None:
            all_words += self._lex.vocabulary
        all_words += self._tfidf.vocabulary_
        return list(set(all_words))

    def _get_properties(self) -> dict:
        properties = {
            "model_name": self.__class__.__name__,
            "use_ctfidf": self._use_ctfidf,
            "random_state": self._random_state,
            "progress_bar": self._use_progress_bar,
            "relevant_pos": self._relevant_pos,
            "min_df": self._min_df,
            "max_df": self._max_df,
            "spacy_model": self._spacy_model,
            "n_docs": self._n_docs,
            "corpus_path": self._corpus_path,
            "corpus_path_encoding": self._corpus_path_encoding,
            "load_clustering": self._load_clustering,
            "checkterm": self._checkterm,
            "n_top_clusters": self._n_top_clusters,
            "cluster_share": self._cluster_share,
            "clustermethod": self._clustermethod,
            "distance_threshold": self._distance_threshold,
            "n_words": self._n_words,
        }
        return properties

    def _set_properties(self, properties: dict) -> None:
        self._use_ctfidf = properties["use_ctfidf"]
        self._random_state = properties["random_state"]
        self._use_progress_bar = properties["progress_bar"]
        self._relevant_pos = properties["relevant_pos"]
        self._min_df = properties["min_df"]
        self._max_df = properties["max_df"]
        self._spacy_model = properties["spacy_model"]
        self._n_docs = properties["n_docs"]
        self._corpus_path = properties["corpus_path"]
        self._corpus_path_encoding = properties["corpus_path_encoding"]
        self._load_clustering = properties["load_clustering"]
        self._checkterm = properties["checkterm"]
        self._n_top_clusters = properties["n_top_clusters"]
        self._cluster_share = properties["cluster_share"]
        self._clustermethod = properties["clustermethod"]
        self._distance_threshold = properties["distance_threshold"]
        self._n_words = properties["n_words"]

    def _set_progress_bar(self):
        if self._use_progress_bar:
            return tqdm
        return self._emptyfunc

    @staticmethod
    def _emptyfunc(array):
        return array

    @staticmethod
    def _make_embeddings(
        embeds: Union[embeddings.Embeddings, dict], checkterm: str
    ) -> embeddings.Embeddings:
        if not isinstance(embeds, embeddings.Embeddings):
            my_embeds = embeddings.Embeddings(
                embedding_dict=embeds, testvalue=checkterm
            )
        else:
            my_embeds = embeds
        return my_embeds

    def _filter_embeddings(self) -> None:
        vocab = self._get_full_vocab()
        self._embeddings.filter_terms(vocab)

    @staticmethod
    def _check_zippath(path: str) -> str:
        if not path.endswith(".weelex.zip"):
            if path.endswith(".weelex"):
                usepath = path + ".zip"
            else:
                usepath = path + ".weelex.zip"
        else:
            usepath = path
        return usepath

    # ---------------------------------------------------------------------------
    # properties:
    @property
    def vocabulary(self) -> List[str]:
        """List of words that are considered by the model.

        Returns:
            List[str]: Words in the vocabulary.

        """
        return list(set(self._lex.vocabulary + self._tfidf.vocabulary_))

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors.

        Returns:
            int: Number of dimensions.

        """
        return self._embeddings.dim

    @property
    def is_fit(self) -> bool:
        """Tells whether the model has been fitted already

        Returns:
            bool: Model is fitted.
        """
        return self._is_fit

    # ---------------------------------------------------------------------------
    # classmethods:
    @classmethod
    def load(cls, path: str) -> "_BasePredictor":
        """Method to load a previously saved instance from disk.

        Args:
            path (str): Location of saved model.

        Returns:
            _BasePredictor: previously saved instance of the model.
        """
        instance = cls(embeds=None)
        usepath = cls._check_zippath(path)
        with ZipFile(usepath) as myzip:
            with myzip.open("properties.json", "r") as f:
                properties = json.load(f)
            instance._set_properties(properties=properties)

            with myzip.open("embeddings.npz", "r") as f:
                instance._embeddings = embeddings.Embeddings.load_filtered(f)
            instance._load_predictprocessor(myzip)
            instance._lex = lexicon.load(path="lex/", archive=myzip)

        instance._is_fit = True
        return instance
