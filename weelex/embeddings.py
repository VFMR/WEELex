from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
import pandas as pd
from gensim.models.fasttext import load_facebook_vectors
from gensim.models import FastText
from gensim.models import Word2Vec



@dataclass
class Embeddings:
    def __init__(self, inputs=None) -> None:
        # if isinstance(inputs, dict):
        #     # TODO: make sure that self._keys and self._vectors would not appear in fields
        #     self._keys: ClassVar, self._vectors: ClassVar = self._data_from_dct(inputs)
        self.isfiltered = False
        self._keys = None
        self._vectors = None
        self._wv = None


    # TODO: how do I handle these well?
    def load_facebook_vectors(self, path):
        wv = load_facebook_vectors(path)
        self._wv = wv


    def load_finetuned_fasttext(self, path):
        wv = FastText.load(path).wv
        self._wv = wv


    def load_finetuned_word2vec(self, path):
        wv = Word2Vec.load(path).wv
        self._wv = wv


    def filter_terms(self, terms: Union[list, np.ndarray, pd.DataFrame, tuple]) -> None:
        if self.isfiltered:
            myvecs = self._vectors
        else:
            myvecs = self._wv

        keys, vectors = self._data_from_dct(
            {term: myvecs[term] for term in terms}
        )
        self._keys = keys
        self._vectors = vectors
        self.isfiltered = True
        self._wv = None


    @staticmethod
    def _data_from_dct(dct:dict) -> Tuple[list, np.array]:
        testvalue = list(dct.values())[0]
        vectors = np.zeros( (len(dct), len(testvalue)) )
        keys = np.array(list(dct.keys()))
        for i, key in enumerate(keys):
            vectors[i] = dct[key]
        return keys, vectors


    def _get_val_from_key(self, key):
        if self.isfiltered:
            index = self._keys.tolist().index(key)
            result =  self._vectors[index, :]
        else:
            result = self._wv[key]
        return result


    def _get_val_from_key_vectorized(self, keys):
        # TODO: currently only working for filtered embeddings
        # TODO: is not working efficiently right now. Only basic functionality
        # sorter = np.argsort(keys)
        # values = sorter[np.searchsorted(self._vectors, keys, sorter=sorter)]
        lst = [self._get_val_from_key(x) for x in keys]
        values = np.array(lst)
        return values


    def save_filtered(self, path: str) -> None:
        if self._keys is not None:
            try:
                np.savez_compressed(path, keys=self._keys, vectors=self._vectors)
            except Exception as e:
                print('File could not be saved')
        else:
            raise ValueError('There are no key-vector pairs to be saved yet. Please apply the filter() method before.')


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


    def keys(self):
        return self._keys


    def lookup(self, terms: Union[str, list, np.ndarray, pd.Series]):
        if isinstance(terms, str):
            values =  self._get_val_from_key(terms)
        else:
            if not isinstance(terms, np.ndarray):
                terms = np.array(terms)
            values = self._get_val_from_key_vectorized(terms)
        return values


    def __getitem__(self, key):
        return self._get_val_from_key(key)
