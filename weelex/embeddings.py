from dataclasses import dataclass
from typing import Union, Tuple, ClassVar

import numpy as np


@dataclass
class Embeddings:
    def __init__(self, inputs: dict) -> None:
        # if isinstance(inputs, dict):
        #     # TODO: make sure that self._keys and self._vectors would not appear in fields
        #     self._keys: ClassVar, self._vectors: ClassVar = self._data_from_dct(inputs)
        pass

    def filter_terms(self, terms: list) -> None:
        pass


    def _data_from_dct(dct:dict) -> Tuple[list, np.array]:
        testvalue = dct.values()[0]
        vectors = np.zeros( (len(dict, len(testvalue))) )
        keys = np.array(dct.keys())
        for i, key in enumerate(keys):
            vectors[i] = dct[key]
        return keys, vectors


    def _get_val_from_key(self, key):
        index = self._keys().tolist().index(key)
        return self._vectors[index, :]
