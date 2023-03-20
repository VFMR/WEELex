"""Contains the class for embedding vectors.
"""
from typing import Union, Tuple, Iterable, overload, List, Dict
from zipfile import ZipFile

import numpy as np
import pandas as pd
from gensim.models.fasttext import load_facebook_vectors
from gensim.models import FastText
from gensim.models import Word2Vec

from weelex import lexicon


class Embeddings:
    """Class for embedding vector lookup."""

    def __init__(
        self, embedding_dict: Dict[str, float] = None, testvalue="test"
    ) -> None:
        """
        Args:
            embedding_dict (Dict[str, float], optional): Embedding vector
                lookup object. Defaults to None.
            testvalue (str, optional): Word that is tested against the
                embedding vectors. Required for example to infer the
                dimensionality. Make sure it is included among the provided
                embedding vectors. Defaults to "test".
        """
        self.isfiltered = False
        self._testvalue = testvalue
        if embedding_dict is not None:
            self._wv = embedding_dict
            self._keys = embedding_dict.keys()
            self._vectors = self._get_val_from_key_vectorized(self._keys)
        else:
            self._wv = None
            self._keys = None
            self._vectors = None

    def filter_terms(
        self, terms: Union[list, lexicon.Lexicon, np.ndarray, pd.DataFrame, tuple]
    ) -> None:
        """Filter the embeddings down to a word-vector collection that is
        required, i.e., is included in the vocabulary. Since WEELex utilizes
        only a fixed number of terms, removing unnecessary ones preserves disk
        memory when saving the model.

        Args:
            terms (Union[list, lexicon.Lexicon, np.ndarray, pd.DataFrame, tuple]):
                Reduce the embedding lookup to these terms
        """
        if isinstance(terms, np.ndarray) or isinstance(terms, pd.DataFrame):
            terms_lst = self._flatten_matrix(terms)
        elif isinstance(terms, lexicon.Lexicon):
            terms_lst = terms.vocabulary
        else:
            terms_lst = terms

        keys, vectors = self._data_from_dct(
            {term: self._get_val_from_key(term) for term in terms_lst}
        )
        self._keys = keys
        self._vectors = vectors
        self.isfiltered = True
        self._wv = self.make_wv_from_keys_vectors()

    def make_wv_from_keys_vectors(self) -> Dict[str, np.ndarray]:
        """Create word-vector lookup.

        Returns:
            Dict[str, np.ndarray]: Word - embedding vector mapping.
        """
        return {key: value for key, value in zip(self._keys, self._vectors)}

    @staticmethod
    def _flatten_matrix(mat: Union[np.ndarray, pd.DataFrame]) -> List[str]:
        lst = []
        append = lst.append
        if isinstance(mat, np.ndarray):
            for row in mat:
                for cell in row:
                    append(cell)
        elif isinstance(mat, pd.DataFrame):
            for col in mat.columns:
                for cell in mat[col]:
                    append(cell)
        lst = [x for x in set(lst) if not np.isnan(x)]
        return lst

    @staticmethod
    def _data_from_dct(dct: dict) -> Tuple[list, np.array]:
        testvalue = list(dct.values())[0]
        vectors = np.zeros((len(dct), len(testvalue)))
        keys = np.array(list(dct.keys()))
        for i, key in enumerate(keys):
            vectors[i] = dct[key]
        return keys, vectors

    def _get_val_from_key(self, key: str) -> np.ndarray:
        if self.isfiltered:
            try:
                index = self._keys.tolist().index(key)
                result = self._vectors[index, :]
            except ValueError as e:
                result = np.zeros((self.dim))
        else:
            result = self._wv[key]
        return result

    def _get_val_from_key_vectorized(self, keys: Iterable[str]) -> np.ndarray:
        # TODO: currently only working for filtered embeddings
        # TODO: is not working efficiently right now. Only basic functionality
        vector = np.zeros(shape=(len(keys), self.dim))
        for i, x in enumerate(keys):
            vector[i] = self._get_val_from_key(x)
        return vector

    def save_filtered(self, path: str) -> None:
        """Save the filtered embeddings to disk.

        Args:
            path (str): Where the vectors are saved.

        Raises:
            ValueError: if the embedding vectors have not been filtered with
                the `filter_terms` method before.
        """
        if self._keys is not None:
            try:
                np.savez_compressed(path, keys=self._keys, vectors=self._vectors)
            except Exception as e:
                print("File could not be saved")
        else:
            raise ValueError(
                """There are no key-vector pairs to be saved yet.
                    Please apply the filter_terms method before."""
            )

    @property
    def keys(self) -> List[str]:
        """Words with available embedding vectors.

        Returns:
            List[str]: Words
        """
        return self._keys

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors

        Returns:
            int: Dimensionality
        """
        try:
            result = len(self._wv[self._testvalue])
        except:
            result = len(self._vectors[0])
        return result

    @overload
    def lookup(self, terms: str) -> np.ndarray:
        ...

    @overload
    def lookup(self, terms: Union[list, np.ndarray, pd.Series]) -> np.ndarray:
        ...

    def lookup(self, terms: Union[str, list, np.ndarray, pd.Series]) -> np.ndarray:
        """Retrieve the embedding vectors for a single word or multiple terms.

        Args:
            terms (Union[str, list, np.ndarray, pd.Series]): Single word or
                iterable of words.

        Returns:
            np.ndarray: Array with the embedding vectors. Has size `dim` for a
                single term and `(number words, dim)` for an interable of words.
        """
        if isinstance(terms, str):
            values = self._get_val_from_key(terms)
        else:
            if not isinstance(terms, np.ndarray):
                terms = np.array(terms)
            values = self._get_val_from_key_vectorized(terms)
        return values

    def __getitem__(self, key: str) -> np.ndarray:
        return self._get_val_from_key(key)

    # --------------------------------------------------------------------------
    # classmethods
    @classmethod
    def load_filtered(cls, path: str, archive: ZipFile = None) -> "Embeddings":
        """Load previously filtered and saved Embeddings object from disk.

        Args:
            path (str): Location of saved object.
            archive (ZipFile, optional): Name of zip archive the object is in.
                Defaults to None.

        Raises:
            ValueError: If the provided object is not supported.

        Returns:
            Embeddings: Loaded instance
        """
        instance = cls()
        if archive is None:
            if isinstance(path, str):
                if not path.endswith(".npz"):
                    path = path + ".npz"
            with np.load(path, allow_pickle=False) as loaded:
                if isinstance(loaded["keys"], np.ndarray):
                    instance._keys = loaded["keys"]
                    failure = False
                else:
                    failure = True

                if isinstance(loaded["vectors"], np.ndarray):
                    instance._vectors = loaded["vectors"]
                else:
                    failure = True
        else:
            if isinstance(path, str):
                if not path.endswith(".npz"):
                    path = path + ".npz"
            with archive.open(path, "r") as f:
                with np.load(f, allow_pickle=False) as loaded:
                    if isinstance(loaded["keys"], np.ndarray):
                        instance._keys = loaded["keys"]
                        failure = False
                    else:
                        failure = True

                    if isinstance(loaded["vectors"], np.ndarray):
                        instance._vectors = loaded["vectors"]
                    else:
                        failure = True

        if failure:
            raise ValueError("Wrong data type in loaded file.")

        instance.isfiltered = True
        instance._wv = instance.make_wv_from_keys_vectors()
        return instance

    @classmethod
    def load_vectors(
        cls, path: str, embedding_type: str, fine_tuned: bool = False
    ) -> "Embeddings":
        """Instantiate Embeddings with pre-trained FastText or Word2Vec models.

        Args:
            path (str): Location of the embeddings to load.
            embedding_type (str): The type of embedding vectors. One of
                {fasttext, word2vec}.
            fine_tuned (bool, optional): Whether the embedding vectors have
                been fine tuned. Matters if `embedding_type='fasttext'` since
                self-trained or fine tuned FastText models need to be read
                differently than the official ones from [https://fasttext.cc/].

        Example:
            >>> embeds = Embeddings.load_vectors('path/to/embeddings')  # doctest: +SKIP
            ...

        Raises:
            ValueError: If an invalid `embedding_type` is entered.

        Returns:
            Embeddings: Embeddings object.
        """
        if embedding_type == "fasttext":
            if not fine_tuned:
                inst = cls._load_facebook_vectors(path)
            else:
                inst = cls._load_finetuned_fasttext(path)
        elif embedding_type == "word2vec":
            inst = cls._load_finetuned_word2vec(path)
        else:
            raise ValueError(
                f"""
                Invalid embedding type {embedding_type}.
                Enter one of 'word2vec' or 'fasttext'.
             """
            )
        return inst

    @classmethod
    def _load_facebook_vectors(cls, path: str) -> None:
        inst = cls()
        wv = load_facebook_vectors(path)
        inst._wv = wv
        return inst

    @classmethod
    def _load_finetuned_fasttext(cls, path: str) -> None:
        inst = cls()
        wv = FastText.load(path).wv
        inst._wv = wv
        return inst

    @classmethod
    def _load_finetuned_word2vec(cls, path: str) -> None:
        inst = cls()
        wv = Word2Vec.load(path).wv
        inst._wv = wv
        return inst
