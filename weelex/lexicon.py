from dataclasses import dataclass
from typing import Union, Iterable, Tuple, Dict
import os
import json
import copy
from zipfile import ZipFile

import pandas as pd
import numpy as np


@dataclass
class BaseLexicon:
    # TODO: Allow for train_test_split to work
    def __init__(
        self,
        dictionary: Union[dict, str, pd.DataFrame],
        sep: str = None,
        encoding: str = None,
    ):
        self._embeddings = None
        if dictionary is not None:
            self._dictionary_df = self._build_dictionary(dictionary, sep, encoding)

    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def copy(self):
        obj = copy.copy(self)
        return obj

    def _build_dictionary(
        self,
        dictionary: Union[dict, str, pd.DataFrame],
        sep: str = None,
        encoding: str = None,
    ) -> pd.DataFrame:
        """Processes the input lexicon and returns it in a pandas DataFrame.

        Args:
            dictionary (dict, str or pd.DataFrame): The key-value pairs to use
                for the lexicon. If str is passed, it should be the path to
                a csv (containing tabular data) or json (containig key-value
                pairs) file.
                If the file ends with ".json", it will attempt to read the
                file with the json module. Otherwise pd.read_csv is attempted.
            sep (str, optional): separator character when reading csv file

        Example:
            >>> my_dct = {'A': ['a', 'b'], 'B': ['c', 'd']}
            >>> l = BaseLexicon(my_dct)
            >>> l._build_dictionary(my_dct)
               A  B
            0  a  c
            1  b  d

        Returns:
            pd.DataFrame: lexicon matrix
        """
        if isinstance(dictionary, str):
            if dictionary.endswith(".json"):
                with open(dictionary, "r") as f:
                    my_dct = json.loads(f.read())
                    dct_df = self._dict_to_df(my_dct)
            else:
                dct_df = pd.read_csv(dictionary, sep=sep, encoding=encoding)
        elif isinstance(dictionary, pd.DataFrame):
            dct_df = dictionary
        elif isinstance(dictionary, dict):
            dct_df = self._dict_to_df(dictionary)

        return dct_df

    @staticmethod
    def _dict_to_df(dictionary):
        return pd.DataFrame(_dict_padding(dictionary))

    @staticmethod
    def _clean_strings(array: pd.Series) -> pd.Series:
        """Replace the '*'-symbols from terms that may be used for
        matching different terms. These are not meaningful for the
        embedding method.

        Example:
            >>> array = pd.Series(['Test*', 'Te*st', '*test'])
            >>> BaseLexicon._clean_strings(array)
            0    Test
            1    Test
            2    test
            dtype: object

        Args:
            array (pd.Series): pandas Series containing strings

        Returns:
            pd.Series: Array without stars
        """
        return array.str.replace("*", "", regex=False)

    @staticmethod
    def _embed_word(term, embeddings):
        if term is not None and term is not np.nan:
            result = embeddings[term]
        else:
            empty_array = np.empty(shape=(embeddings.dim))
            result = empty_array.fill(np.nan)
        return result

    @staticmethod
    def _nonmissarray(array: pd.Series) -> pd.Series:
        """

        Example:
            >>> array = pd.Series(['one', 'two', 'three', np.nan, np.nan])
            >>> BaseLexicon._nonmissarray(array)
            0      one
            1      two
            2    three
            dtype: object

        Args:
            array (pd.Series): Array to be get elements that are not nan

        Returns:
            pd.Series: Array containing only non-nan elements
        """
        nonmiss = array[~array.isna()]
        return nonmiss

    def embed(self, embeddings) -> None:
        dict_df_shape = self._dictionary_df.shape
        embedding_tensor = np.zeros(
            shape=(dict_df_shape[0], dict_df_shape[1], embeddings.dim)
        )
        for j, key in enumerate(self._dictionary_df.columns):
            for i, x in enumerate(self._dictionary_df[key]):
                embedding_tensor[i][j] = self._embed_word(x, embeddings)

        self._embeddings = embedding_tensor

    def get_vocabulary(self) -> list:
        """Returns a sorted list of all the lexicon categories

        Example:
            >>> my_dct = {'Food': ['Bread', 'Salad'], 'Animal': ['Dog', 'Cat']}
            >>> l = BaseLexicon(my_dct)
            >>> l.get_vocabulary()
            ['Bread', 'Cat', 'Dog', 'Salad']

        Returns:
            list: Sorted array of categories
        """
        vocab = []
        for col in self._dictionary_df:
            vocab += list(self._dictionary_df[col])
        vocab = [x for x in vocab if isinstance(x, str)]  # removal of np.nans
        return sorted(list(set(vocab)))

    def _append_values(self, lex: "Lexicon", key: str, values: pd.Series) -> pd.Series:
        maxlen = lex._dictionary_df.shape[0]
        collen = len(lex._dictionary_df[~lex._dictionary_df[key].isna()])
        # TODO: Implement rest of method _append_values()

    def to_dict(self) -> dict:
        """Return the lexicon in dictionary format.

        Example:
            >>> my_dct = {'A': ['a', 'b'], 'B': ['c', 'd']}
            >>> l = BaseLexicon(my_dct)
            >>> l.to_dict()
            {'A': ['a', 'b'], 'B': ['c', 'd']}

        Returns:
            dict: dict with the lexicon key-value pairs
        """
        out_dict = {
            key: list(self._dictionary_df[key]) for key in self._dictionary_df.columns
        }
        return out_dict

    def _get_properties(self):
        properties = {"name": self.__class__.__name__}
        return properties

    def save(self, path: str) -> None:
        """Save lexicon to disk

        Args:
            path (str): Output file path
        """
        os.makedirs(path, exist_ok=True)
        self._dictionary_df.to_csv(
            os.path.join(path, "dictionary_df.csv.gz"), compression="gzip", index=False
        )

        np.save(os.path.join(path, "embeddings"), self._embeddings, allow_pickle=False)

        with open(os.path.join(path, "properties.json"), "w") as f:
            json.dump(self._get_properties(), f)

    # ------------------------------- Properties

    @property
    def keys(self):
        return list(self._dictionary_df.columns)

    @property
    def embedding_shape(self):
        return self._embeddings.shape

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def vocabulary(self) -> list:
        return self.get_vocabulary()

    @property
    def is_embedded(self) -> bool:
        return self._embeddings is not None

    # ------------------------------ Class methods

    @classmethod
    def load(cls, path: str, properties: dict = None, archive: ZipFile = None):
        """Load a previously saved Lexicon instance

        Args:
            path (str): Path of saved Lexicon instance
        """
        if properties is None:
            if archive is None:
                with open(os.path.join(path, "properties.json"), "r") as f:
                    properties = json.load(f)
            else:
                with archive.open(path + "properties.json", "r") as f:
                    properties = json.load(f)
        propertyname = properties["name"]
        if propertyname != cls.__name__:
            raise ValueError(
                f"Attempted to load a saved {propertyname} instance when a {cls.__name__} instance is required."
            )

        if archive is None:
            dictionary_df = pd.read_csv(os.path.join(path, "dictionary_df.csv.gz"))
            embeddings = np.load(os.path.join(path, "embeddings.npy"))
        else:
            with archive.open(path + "dictionary_df.csv.gz", "r") as f:
                try:
                    dictionary_df = pd.read_csv(f, compression="gzip")
                except UnicodeDecodeError:
                    dictionary_df = pd.read_csv(
                        f, encoding="latin1", compression="gzip"
                    )
            with archive.open(path + "embeddings.npy") as f:
                embeddings = np.load(f)

        instance = cls(dictionary=None)
        instance._embeddings = embeddings
        instance._dictionary_df = dictionary_df
        return instance


class WeightedLexicon(BaseLexicon):
    def __init__(
        self,
        dictionary: Union[dict, str, pd.DataFrame],
        sep: str = None,
        encoding: str = None,
    ) -> None:
        super().__init__(dictionary=dictionary, sep=sep, encoding=encoding)
        if dictionary is not None:
            df, weights = self._build_weighted_dictionary(self._dictionary_df)
            self._dictionary_df = df
            self._weights = weights
            self._word2weight = self._get_word2weight()

    def __getitem__(self, key: str) -> Union[float, int]:
        """getter for simple data retrieval

        Example:
            >>> lex = WeightedLexicon({'a': 0.1, 'b': 0.2})
            >>> lex['a']
            0.1

        Args:
            key (str): Term

        Returns:
            Uniont[float, int]: weight value
        """
        return self._get_word_value(key)

    def __repr__(self) -> Dict[str, Union[float, int]]:
        """repr

        Example:
            >>> lex = WeightedLexicon({'a': 0.1, 'b': 0.2})
            >>> lex
            {'a': 0.1, 'b': 0.2}

        Returns:
            dict: Dictionary with word-weight pairs.
        """
        return (
            "{"
            + ", ".join(
                [f"'{word}': {weight}" for word, weight in self._word2weight.items()]
            )
            + "}"
        )

    @staticmethod
    def _dict_to_df(dict):
        return pd.DataFrame([[key, value] for key, value in dict.items()])

    @staticmethod
    def _build_weighted_dictionary(
        raw_dict: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """The dictionary is expected to be a key-value pair with of the form
        {word: weight} or a table or DataFrame where the first column
        contains the words and the second one contains the scores.
        This method splits the table into a matrix with words and an array
        of weights. Words are contained in a single column DataFrame to ensure
        compatibility with the base class.

        Args:
            raw_dict (pd.DataFrame): result from the _build_dictionary method

        Example:
            >>> my_dct = pd.DataFrame({0: ['a', 'b'], 1: [0.1, 0.2]})
            >>> x, y = WeightedLexicon._build_weighted_dictionary(my_dct)
            >>> print(x)
               0
            0  a
            1  b
            >>> print(y)
            0    0.1
            1    0.2
            Name: 1, dtype: float64

        Returns:
            (pd.DataFrame, pd.Series): lexicon matrix
        """
        dictionary_df = pd.DataFrame(raw_dict.iloc[:, 0])
        weights = raw_dict.iloc[:, 1]
        return dictionary_df, weights

    def _get_word2weight(self) -> Dict[str, Union[float, int]]:
        words = list(self._dictionary_df.iloc[:, 0])
        weights = list(self._weights)
        return {x: y for x, y in zip(words, weights)}

    def _get_word_value(self, word: str) -> Union[float, int]:
        return self._word2weight[word]

    def to_dict(self) -> Dict[str, Union[float, int]]:
        return self._word2weight

    def embed(self, embeddings) -> None:
        super().embed(embeddings)
        self._embeddings = self._embeddings[:, 0, :]

    def _get_properties(self):
        properties = super()._get_properties()
        properties.update({"_word2weight": self._word2weight})
        return properties

    def save(self, path):
        super().save(path)
        self._weights.to_csv(
            os.path.join(path, "weights.csv.gz"), compression="gzip", index=False
        )

    # ---------------------------------------------------------------------------
    # Properties:
    @property
    def weights(self):
        return self._weights

    @property
    def vocabulary(self) -> list:
        # overwrite such that order is unchanged
        return list(self._dictionary_df.iloc[:, 0])

    # ---------------------------------------------------------------------------
    # Classmethods:
    @classmethod
    def load(cls, path, properties, archive: ZipFile = None):
        instance = super().load(path, properties, archive=archive)
        if archive is None:
            weights = pd.read_csv(os.path.join(path, "weights.csv.gz")).iloc[:, 0]
        else:
            with archive.open(path + "weights.csv.gz") as f:
                try:
                    weights = pd.read_csv(f, compression="gzip").iloc[:, 0]
                except UnicodeDecodeError:
                    weights = pd.read_csv(
                        f, encoding="latin1", compression="gzip"
                    ).iloc[:, 0]
        instance._weights = weights
        instance._word2weight = properties["_word2weight"]
        return instance


class Lexicon(BaseLexicon):
    def __init__(
        self,
        dictionary: Union[dict, str, pd.DataFrame],
        sep: str = None,
        encoding: str = None,
    ):
        super().__init__(dictionary=dictionary, sep=sep, encoding=encoding)

    def __getitem__(self, key: str) -> pd.Series:
        """getter for simple data retrieval

        Example:
            >>> lex = Lexicon({'A': ['a', 'b'], 'B': ['c', 'd', 'e']})
            >>> lex['A']
            0      a
            1      b
            2    NaN
            Name: A, dtype: object

        Args:
            key (str): Lexicon key/category

        Returns:
            pd.Series: Array containing the words
        """
        return self._dictionary_df[key]

    def __setitem__(self, key: str, value: Iterable[str]) -> None:
        """setter method to add lexicon keys/categories

        Example:
            >>> lex = Lexicon({'A': ['a', 'b']})
            >>> lex['B'] = ['c', 'd']
            >>> print(lex._dictionary_df)
               A  B
            0  a  c
            1  b  d

        Args:
            key (str): Name of the category
            value (Iterable[str]): Array containing the terms
        """
        self._dictionary_df[key] = value

    def __repr__(self):
        """repr

        Example:
            >>> lex = Lexicon({'A': ['a', 'b'], 'B': ['c', 'd']})
            >>> lex
               A  B
            0  a  c
            1  b  d

        Returns:
            pd.DataFrame: Matrix of categories and terms
        """
        return self._dictionary_df.__repr__()

    def merge(
        self,
        lexica: Union["BaseLexicon", Iterable["BaseLexicon"]],
        inplace: bool = True,
    ) -> None:
        if inplace:
            obj = self
        else:
            obj = self.copy()

        if isinstance(lexica, Lexicon):
            obj._merge_one(lexica)
        else:
            for lex in lexica:
                obj._merge_one(lex)

        if not inplace:
            return obj

    def _merge_one(self, lex: "BaseLexicon") -> None:
        old_dct = self._dictionary_df.copy()
        old_keys = old_dct.keys()
        new_keys = lex.keys
        update_keys = [x for x in new_keys if x in old_keys]
        append_keys = [x for x in new_keys if x not in old_keys]
        new_dct = pd.concat(
            [old_dct, lex._dictionary_df.loc[:, append_keys]],
            # ignore_index=True,
            axis=1,
        )
        self._dictionary_df = new_dct
        # TODO: allow for merge of existing keys


def _dict_padding(dictionary: dict, filler=np.nan) -> dict:
    """Padding of a dictionary where the values are lists such that these lists
    have the same length.

    Args:
        dictionary (dict): Dictionary where the values are lists,
            e.g. {'A': [1,2,3], 'B': [1,2]}
        filler: Value to pad with. Defaults to np.nan

    Example:
        >>> _dict_padding({'A': [1,2,3], 'B': [1,2]})
        {'A': [1, 2, 3], 'B': [1, 2, nan]}

    Returns:
        dict: Padded dictionary
    """
    lenghts = [len(dictionary[x]) for x in dictionary.keys()]
    maxlen = max(lenghts)
    padded_dict = {}
    for key, value in dictionary.items():
        new_list = _list_padding(lst=value, maxlen=maxlen, filler=filler)
        padded_dict.update({key: new_list})
    return padded_dict


def _list_padding(lst: list, maxlen: int, filler=np.nan) -> list:
    """Appends a filler value to a list such that the list has the preferred
    length, or cut values at the end

    Args:
        lst (list): List to fill
        maxlen (int): Desired length
        filler (optional): Value to fill list with. Defaults to np.nan.

    Example:
        >>> _list_padding(lst=[1,2], maxlen=4, filler=0)
        [1, 2, 0, 0]
        >>> _list_padding(lst=[1,2,3], maxlen=2, filler=0)
        [1, 2]
        >>> _list_padding(lst=[1,2,3], maxlen=5, filler=np.nan)
        [1, 2, 3, nan, nan]

    Returns:
        list: padded list
    """
    len_diff = maxlen - len(lst)
    new_lst = lst.copy()
    if len_diff > 0:
        for _ in range(len_diff):
            new_lst.append(filler)
    if len_diff < 0:
        new_lst = new_lst[0:maxlen]
    return new_lst


def merge_lexica(lexica: Iterable[Lexicon]) -> Lexicon:
    lex1 = lexica[0]
    if len(lexica) > 1:
        for lex in lexica[1:]:
            lex1.merge(lex)
    return lex1


def load(
    path: str,
    archive: ZipFile = None,
) -> Union["BaseLexicon", "WeightedLexicon", "Lexicon"]:
    if archive is None:
        with open(os.path.join(path, "properties.json"), "r") as f:
            properties = json.load(f)
    else:
        if path.endswith("/"):
            filepath = path + "properties.json"
        else:
            filepath = path + "/properties.json"
        with archive.open(filepath, "r") as f:
            properties = json.load(f)

    propertyname = properties["name"]
    if propertyname == "BaseLexicon":
        instance = BaseLexicon.load(path, properties, archive=archive)
    elif propertyname == "WeightedLexicon":
        instance = WeightedLexicon.load(path, properties, archive=archive)
    elif propertyname == "Lexicon":
        instance = Lexicon.load(path, properties, archive=archive)
    else:
        raise ValueError(
            f"Attempted to load an instance of {propertyname} when one of"
            + ' {"BaseLexicon", "WeightedLexicon", "Lexicon"} was expected.'
        )
    return instance
