"""
Contains the main Weelex classifier class.
"""
from typing import Union, Iterable, Dict, List
import os
from zipfile import ZipFile

from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
import batchprocessing
from cluster_tfidf.ctfidf import ClusterTfidfVectorizer

from weelex import lexicon
from weelex import embeddings
from weelex import _ensemble
from weelex import _base
from weelex._trainer import _TrainProcessor
from weelex.tfidf import BasicTfidf


class WEELexClassifier(_base._BasePredictor):
    """Main Weelex classifier class"""

    def __init__(
        self,
        embeds: Union[dict, embeddings.Embeddings],
        tfidf: Union[str, BasicTfidf] = None,
        ctfidf: Union[str, ClusterTfidfVectorizer] = None,
        use_ctfidf: bool = True,
        word_level_aggregation: bool = True,
        test_size: float = None,
        random_state: int = None,
        n_jobs: int = 1,
        progress_bar: bool = False,
        relevant_pos: List[str] = ["ADJ", "ADV", "NOUN", "VERB"],
        min_df: Union[int, float] = 5,
        max_df: Union[int, float] = 0.95,
        spacy_model: str = "de_core_news_lg",
        n_docs: int = 2000000,
        corpus_path: str = None,
        corpus_path_encoding: str = "latin1",
        load_clustering: bool = False,
        checkterm: str = "Politik",
        n_top_clusters: int = 3,
        cluster_share: float = 0.2,
        clustermethod: str = "agglomerative",
        distance_threshold: float = 0.5,
        n_words: int = 40000,
        **train_params,
    ) -> None:
        """Class initialization

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
            use_ctfidf (bool, optional): Whether to use Cluster Tfidf or not.
                Defaults to True.
            word_level_aggregation (bool, optional): Whether the output
                of document level inputs shall be aggregated on a word level.
                Defaults to True.
            test_size (float, optional): Process only random share of data for
                testing purposes. Defaults to None.
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
        """
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
            n_words=n_words,
        )
        self._model = _ensemble._FullEnsemble
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

    def set_params(self, **params) -> None:
        """Set parameters for the model by passing keywords and their values.

        Example:
            >>> embeds = {'dog': [0,0,0], 'cat': [0,0,0]}
            >>> model = WEELexClassifier(embeds=embeds)
            >>> model.get_params()['n_jobs']
            1
            >>> model.set_params(n_jobs=2, min_df=1, n_words=1000)
            >>> model.get_params()['n_jobs']
            2

        """
        trainparams = {}
        for key, value in params.items():
            # CHECKME: check if this is a valid approach
            if key in self.__dict__:
                self.__dict__[key] == value
            else:
                trainparams.update({key: value})
        self._train_params = trainparams
        return self

    def fit(
        self,
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
        progress_bar: bool = False,
    ) -> None:
        """Fit the model instance by both fitting the tfidf and cluster-tfidf
        instances as well as by training the supervised model for word level
        embedding classification.

        Args:
            X (Union[pd.Series, np.ndarray]): Corpus of training data documents.
            lex (Union[lexicon.Lexicon, dict, str]): The dictionary or lexicon
                that is used for the main classes of the model.
            support_lex (Union[lexicon.Lexicon, dict, str], optional):
                Additional lexicon categories and their words. These will not be
                among the predicted classes but can improve performance.
                Defaults to None.
            main_keys (Iterable[str], optional): Names of main categories to
                predict. Can be used if `lex` contains both main categories and
                support categories while `support_lex` is None. Select these
                categories as main categories. Defaults to None.
            support_keys (Iterable[str], optional): Names of support categories.
                 Can be used if `lex` contains both main categories and
                support categories while `support_lex` is None. Select these
                categories as support categories. Defaults to None.
            hp_tuning (bool, optional): Whether hyperparameters shall be tuned.
                Defaults to False.
            n_iter (int, optional): Number of iterations for RandomizedSearchCV
                hyperparameter tuning. Defaults to 150.
            cv (int, optional): Number of folds for cross validation.
                Defaults to 5.
            param_grid (dict, optional): Hyperparameter grid for hyperparameter
                tuning. Defaults to None.
            fixed_params (dict, optional): Hyperparameters required in all
                instances. Defaults to None.
            n_best_params (int, optional): Number of best performing models to
                aggregate. Defaults to 3.
            progress_bar (bool, optional): Show progress bar. Defaults to False.
        """
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
                self._hyperparameter_tuning(
                    cat=cat,
                    n_iter=n_iter,
                    param_grid=param_grid,
                    fixed_params=fixed_params,
                    n_best_params=n_best_params,
                    cv=cv,
                )

            model_params = self._tuned_params.get(cat)
            if fixed_params is not None:
                fixed_params_dct = fixed_params
            else:
                fixed_params_dct = {}
            fixed_params_dct.update({"input_shape": input_shape})

            model = self._model(
                cat,
                categories=self._trainprocessor.main_keys,
                outside_categories=self._trainprocessor.support_keys,
                param_set=model_params,
                **fixed_params_dct,
            )
            model.fit(*self._trainprocessor.feed_cat_Xy(cat=cat, train=True))
            models.update({cat: model})
        self._models = models
        self._is_fit = True

    def _get_full_vocab(self) -> list:
        all_terms = super()._get_full_vocab()
        all_terms += self._support_lex.vocabulary
        return list(set(all_terms))

    def save(self, path) -> None:
        """Save the trained model to disk into a compressed archive.

        Args:
            path (str): path to write the model to.
        """
        self._save_objects_to_dir(path)
        if self._support_lex is not None:
            self._support_lex.save(os.path.join(path, "support_lex"))
        self._clean_save(path)

    def _setup_trainprocessor(
        self,
        lex: Union[lexicon.Lexicon, dict, str],
        support_lex: Union[lexicon.Lexicon, dict, str] = None,
        main_keys: Iterable[str] = None,
        support_keys: Iterable[str] = None,
    ) -> None:
        self._trainprocessor = _TrainProcessor(
            lex=lex,
            support_lex=support_lex,
            main_keys=main_keys,
            support_keys=support_keys,
            embeddings=self._embeddings,
            test_size=self._test_size,
            random_state=self._random_state,
        )
        self._trainprocessor.make_train_test_data()
        self._main_keys = self._trainprocessor.main_keys
        self._support_keys = self._trainprocessor.support_keys

    def _hyperparameter_tuning(
        self, cat, n_iter, param_grid, fixed_params, cv: int = 5, n_best_params=3
    ):
        if param_grid is None:
            raise ValueError("No parameter grid set for tuning.")

        if fixed_params is not None:
            fixed_params_dct = fixed_params
        else:
            fixed_params_dct = {}
        search = RandomizedSearchCV(
            estimator=_ensemble._AugmentedEnsemble(
                cat,
                categories=self._main_keys,
                outside_categories=self._support_keys,
                **fixed_params_dct,
            ),
            param_distributions=param_grid,
            n_iter=n_iter,
            n_jobs=self._n_jobs,
            cv=cv,
            scoring=None,
            random_state=self._random_state,
        )

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
        """Returns the mean test scores from non-missing cross validation results.
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
        test_scores = [
            list(value)
            for key, value in results.items()
            if "test_score" in key
            and not "mean" in key
            and not "std" in key
            and not "rank" in key
        ]
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
        for x, y in zip(ranks, results["params"]):
            if x < n_best_params and len(result_params) < n_best_params:
                result_params.append(y)
        return result_params

    def _probas_to_binary(self, probas, cutoff):
        catpreds_binary = {}
        for cat in self._main_keys:
            if len(probas[cat].shape) == 2:
                pred = (probas[cat][:, 0] >= cutoff).astype(int)
            else:
                pred = (probas[cat] >= cutoff).astype(int)
            catpreds_binary.update({cat: pred})
        return pd.DataFrame(catpreds_binary)

    @batchprocessing.batch_predict
    def predict_words(
        self,
        X: pd.DataFrame,
        cutoff: float = 0.5,
        n_batches: int = None,
        checkpoint_path: str = None,
    ) -> pd.DataFrame:
        """Method for binary word level prediction of a set of words.

        Args:
            X (pd.DataFrame): Array of input words to predict.
            cutoff (float, optional): Predict class 1 if probability > cutoff.
                Defaults to 0.5.
            n_batches (int, optional): If int is set, the prediction is run
                in batches with checkpoints. Defaults to None.
            checkpoint_path (str, optional): For checkpointed batch processing,
                specify a path for the checkpoints. Defaults to None.

        Returns:
            pd.DataFrame: Array with predictions. One column for each category.
        """
        catpreds = self.predict_proba_words(X=X)
        return self._probas_to_binary(catpreds, cutoff=cutoff)

    def predict_proba_words(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Method for word level prediction of a set of words.

        Args:
            X (pd.DataFrame): Array of input words

        Returns:
            Dict[str, pd.DataFrame]: Array with predicted probabilities.
        """
        catpreds = {}
        for cat in self._main_keys:
            model = self._models[cat]
            preds = model.predict_proba(X)
            catpreds.update({cat: preds})
        return catpreds

    @batchprocessing.batch_predict
    def predict_docs(
        self,
        X: pd.DataFrame,
        cutoff: float = 0.5,
        n_batches: int = None,
        checkpoint_path: str = None,
    ) -> pd.DataFrame:
        """Method for binary document level prediction of a set of documents.

        Args:
            X (pd.DataFrame): Array of input documents.
            cutoff (float, optional): Predict class if probability > cutoff.
                Defaults to 0.5.
            n_batches (int, optional): If int is set, the prediction is run
                in batches with checkpoints. Defaults to None.
            checkpoint_path (str, optional): For checkpointed batch processing,
                specify a path for the checkpoints. Defaults to None.

        Returns:
            pd.DataFrame: Array with predicted class probabilities. One column
                for each category.
        """
        preds = self.predict_proba_docs(X=X)
        return self._probas_to_binary(preds, cutoff=cutoff)

    def predict_proba_docs(self, X: pd.DataFrame) -> pd.DataFrame:
        """Method for document level prediction of a set of documents.
        Returns predicted probabilities.

        Args:
            X (pd.DataFrame): Array of input documents

        Raises:
            NotFittedError: When attempting to predict with a model instance
                that has not previously been fitted with the .fit() method.

        Returns:
            pd.DataFrame: Array with predicted class probabilities.
                One column for each category.
        """
        if self._is_fit is False:
            raise NotFittedError(
                f'This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'
            )
        vects = self._predictprocessor.transform(X)
        return self.predict_proba_words(vects)

    # ---------------------------------------------------------------------------
    # properties
    @property
    def main_keys(self) -> List[str]:
        """Main categories to predict.

        Returns:
            List[str]: Main categories
        """
        return self._main_keys

    @property
    def support_keys(self) -> List[str]:
        """Provided support categories that are not being predicted.

        Returns:
            List[str]: Support categories
        """
        return self._support_keys

    @property
    def vocabulary(self) -> List[str]:
        """List of words that are considered by the model.

        Returns:
            List[str]: Words in the vocabulary.
        """
        vocab = super().vocabulary
        if self._support_lex is not None:
            vocab += self._support_lex.vocabulary
        return list(set(vocab))

    # ---------------------------------------------------------------------------
    # classmethods:
    @classmethod
    def load(cls, path: str) -> "WEELexClassifier":
        """Method to load a previously saved instance from disk.

        Args:
            path (str): Location of saved model.

        Returns:
            WEELexClassifier: Previously saved instance of the model.
        """
        instance = super().load(path)
        usepath = cls._check_zippath(path)
        with ZipFile(usepath) as myzip:
            instance._support_lex = lexicon.load(path="support_lex/", archive=myzip)
        return instance
