from typing import Union, Tuple, Iterable, overload, List, Dict
# from functools import singledispatch, singledispathmethod

import numpy as np
import pandas as pd
from gensim.models.fasttext import load_facebook_vectors
from gensim.models import FastText
from gensim.models import Word2Vec

from weelex import lexicon


class Embeddings:
    def __init__(self,
                 embedding_dict: Dict[str, float] = None) -> None:
        self.isfiltered = False
        self.testvalue = 'Test'
        if embedding_dict is not None:
            self._wv = embedding_dict
            self._keys = embedding_dict.keys()
            self._vectors = np.array([embedding_dict[x] for x in self._keys])
        else:
            self._wv = None
            self._keys = None
            self._vectors = None

    def load_facebook_vectors(self, path: str) -> None:
        wv = load_facebook_vectors(path)
        self._wv = wv

    def load_finetuned_fasttext(self, path: str) -> None:
        wv = FastText.load(path).wv
        self._wv = wv

    def load_finetuned_word2vec(self, path: str) -> None:
        wv = Word2Vec.load(path).wv
        self._wv = wv

    def load_vectors(self,
                     path: str,
                     embedding_type: str,
                     fine_tuned: bool = False) -> None:
        if embedding_type == 'fasttext':
            if not fine_tuned:
                self.load_facebook_vectors(path)
            else:
                self.load_finetuned_fasttext(path)
        elif embedding_type == 'word2vec':
            self.load_finetuned_word2vec(path)
        else:
            raise ValueError(f"""
                Invalid embedding type {embedding_type}.
                Enter one of 'word2vec' or 'fasttext'.
             """)

    def filter_terms(self,
                     terms: Union[list,
                                  lexicon.Lexicon,
                                  np.ndarray,
                                  pd.DataFrame,
                                  tuple]) -> None:
        if self.isfiltered:
            myvecs = self._vectors
        else:
            myvecs = self._wv

        if isinstance(terms, np.ndarray) or isinstance(terms, pd.DataFrame):
            terms_lst = self._flatten_matrix(terms)
        elif isinstance(terms, lexicon.Lexicon):
            terms_lst = terms.vocabulary
        else:
            terms_lst = terms

        keys, vectors = self._data_from_dct(
            {term: myvecs[term] for term in terms_lst}
        )
        self._keys = keys
        self._vectors = vectors
        self.isfiltered = True
        self._wv = None

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
        lst = [self._get_val_from_key(x) for x in keys]
        values = np.array(lst)
        return values

    def save_filtered(self, path: str) -> None:
        if self._keys is not None:
            try:
                np.savez_compressed(path,
                                    keys=self._keys,
                                    vectors=self._vectors)
            except Exception as e:
                print('File could not be saved')
        else:
            raise ValueError(
                    '''There are no key-vector pairs to be saved yet.
                    Please apply the filter() method before.''')

    def load_filtered(self, path: str) -> None:
        if not path.endswith('.npz'):
            path = path+'.npz'
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded['keys'], np.ndarray):
            self._keys = loaded['keys']
            failure = False
        else:
            failure = True

        if isinstance(loaded['vectors'], np.ndarray):
            self._vectors = loaded['vectors']
        else:
            failure = True

        if failure:
            raise ValueError('Wrong data type in loaded file.')
        self.isfiltered = True

    @property
    def keys(self):
        return self._keys

    @property
    def dim(self):
        try:
            result = len(self._vectors[0])
        except:
            result = len(self._wv)
        return result

    @overload
    def lookup(self, terms: str) -> np.ndarray:
        ...

    @overload
    def lookup(self, terms: Union[list, np.ndarray, pd.Series]) -> np.ndarray:
        ...

    def lookup(self, terms: Union[str, list, np.ndarray, pd.Series]) -> np.ndarray:
        if isinstance(terms, str):
            values = self._get_val_from_key(terms)
        else:
            if not isinstance(terms, np.ndarray):
                terms = np.array(terms)
            values = self._get_val_from_key_vectorized(terms)
        return values

    def __getitem__(self, key: str) -> np.ndarray:
        return self._get_val_from_key(key)
