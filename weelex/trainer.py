from typing import Union, Iterable

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from weelex import lexicon
from weelex import embeddings


class TrainProcessor:
    def __init__(self,
                 lex: lexicon.Lexicon,
                 support_lex: lexicon.Lexicon=None,
                 main_keys: Iterable[str]=None,
                 support_keys: Iterable[str]=None,
                 embeddings: embeddings.Embeddings=None,
                 test_size: Union[float, int]=0.2,
                 random_state: int=None):
        # self._lex = lex
        # self._support_lex = support_lex
        # self._main_keys = main_keys
        # self._support_keys = support_keys
        self._handle_lexica(lex,
                            support_lex,
                            main_keys,
                            support_keys)
        self._embeddings = embeddings
        self._test_size = test_size
        self._random_state = random_state
        
    @staticmethod
    def _make_lexicon(lex: Union[lexicon.Lexicon, dict, str]) -> lexicon.Lexicon:
        """Create proper Lexicon instance from the passed input lexicon

        Args:
            lexicon (dict, str or pd.DataFrame): The key-value pairs to use
                for the lexicon. If str is passed, it should be the path to
                a csv (containing tabular data) or json
                (containig key-value pairs) file.
                If the file ends with ".json", it will attempt to read the
                file with the json module. Otherwise pd.read_csv is attempted.

        Returns:
            lexicon.Lexicon: lexicon.Lexicon instance
        """
        if not isinstance(lex, lexicon.Lexicon):
            my_lex = lexicon.Lexicon(lex)
        else:
            my_lex = lex
        return my_lex

    # REMOVE
    # @staticmethod
    # def _merge_lexica(lexica: Iterable[lexicon.Lexicon]) -> lexicon.Lexicon:
    #     base_lex = lexica[0]
    #     full_lex = base_lex.merge(lexica[1:])
    #     return full_lex
    
    def _handle_lexica(self,
                       main_lex: Union[lexicon.Lexicon, dict, str],
                       support_lex: Union[lexicon.Lexicon, dict, str]=None,
                       main_keys: Iterable[str]=None,
                       support_keys: Iterable[str]=None) -> None:
        # create lexica as Lexicon Instances:
        _main_lex = self._make_lexicon(main_lex)
        if support_lex is not None:
            _support_lex = self._make_lexicon(support_lex)
        else:
            _support_lex = None

        # get keys:
        if main_keys and main_keys is not None:
            _main_keys = main_keys
        else:
            _main_keys = _main_lex.keys

        if support_keys and support_keys is not None:
            _support_keys = support_keys
        elif support_lex is not None:
            _support_keys = _support_lex.keys
        else:
            _support_keys = []

        # Remove any overlapping keys:
        _support_keys = [x for x in _support_keys if x not in _main_keys]

        # Merge lexica:
        # _full_lex = self._merge_lexica([_main_lex, _support_lex])

        # self._lexicon = _full_lex
        self._main_lex = _main_lex
        self._support_lex = _support_lex
        self._main_keys = _main_keys
        self._support_keys = _support_keys

    def _handle_outliers(self):
        # TODO: make method to remove outliers (usually 2 character abbrevs.)
        pass

    @staticmethod
    def _prepare_input_data(lex: lexicon.Lexicon):
        for cat in lex.keys:
            lex[cat] = lex._clean_strings(lex[cat])

        lex.embed()

    def _get_embedding_df(self, lex: lexicon.Lexicon) -> pd.DataFrame:
        if lex.embeddings is None or lex.embeddings.shape[1] != len(lex.keys):
            lex.embed(self._embeddings)
        dfs = []
        for i, cat in enumerate(lex.keys):
            # for j, word in enumerate(self[cat]):
            df = pd.DataFrame(lex.embeddings[:, i])
            # get nomissings:
            catw = lex[cat]
            nomiss = lex._nonmissarray(catw)

            df = df.iloc[:len(nomiss), :]
            df.index = cat+':'+nomiss
            df.index = df.index.str.replace('*', '', regex=False)
            dfs.append(df)
        df_full = pd.concat(dfs)
        return df_full

    @staticmethod
    def _make_id_term_mapping(embedding_df):
        """
        Example:
            >>> embedding_df = pd.DataFrame(np.zeros((3,10)))
            >>> embedding_df.index = ['Cars:Vehicle', 'Politics:Politician', 'Food:Bread']
            >>> x = TrainProcessor(lex={'A': ['a', 'b']})
            >>> x._make_id_term_mapping(embedding_df)
              categories       terms
            0       Cars     Vehicle
            1   Politics  Politician
            2       Food       Bread
        """
        # terms = embedding_df.index.str.replace(r'[A-Za-z0-9]*\:', '',
        #                                        regex=True)
        # term2cat = pd.DataFrame(terms, columns=['terms'])
        # categories = df_full.index.str.extract(r'([A-Za-z0-9]*)\:')
        # term2cat['categories'] = categories
        term2cat = embedding_df.index.str.split(':', expand=True)\
                                         .to_frame(index=False)
        term2cat.columns = ['categories', 'terms']
        return term2cat

    def _prepare_inputs(self):
        lex_full = self._main_lex.merge(self._support_lex, inplace=False)
        embedding_df = self._get_embedding_df(lex_full)
        term2cat = self._make_id_term_mapping(embedding_df)
        # self._lex_full = lex_full
        self._embedding_df = embedding_df
        self._term2cat = term2cat

    def _make_y(self):
        y_classes = {}
        terms_classes = {}

        for cat in self._main_keys:
            # separate terms from the current category and other categories
            cat_terms = self._term2cat[self._term2cat['categories']==cat].loc[:, ['terms']]
            cat_terms[cat] = True

            # "other" terms: words that do not belong to current category
            # but are nonetheless in our list of categories to consider.
            other_terms = self._term2cat[
                (self._term2cat['categories']!=cat) & (self._term2cat['categories'].isin(self._main_keys+self._support_keys))
            ].loc[:, ['terms', 'categories']]

            # handle words that appear in multiple categories: remove from other category
            other_terms = other_terms[~other_terms['terms'].isin(list(cat_terms['terms']))]
            other_terms[cat] = False

            y = pd.concat([cat_terms, other_terms], ignore_index=False, axis=0)
            terms_classes.update({cat: y['terms']})
            del y['terms']
            y_classes.update({cat: y})

        return y_classes

    def _make_X(self, y_classes: dict):
        X_full = self._embedding_df.copy()
        X_full.index = np.arange(len(X_full))
        X_classes = {}
        for cat in self._main_keys:
            X_class_this = X_full.loc[y_classes[cat].index.values, :]
            X_classes.update({cat: X_class_this})
        return X_classes

    def _make_data(self):
        self._prepare_inputs()
        y_classes = self._make_y()
        x_classes = self._make_X(y_classes)
        return x_classes, y_classes

    def _train_test_split(self,
                          X,
                          y,
                          test_size:float=0.2, 
                          random_state:int=None) -> None:
        trains = {}
        tests = {}
        for i, cat in enumerate(self._main_keys):
            _X = X[cat]
            _y = y[cat]
            if test_size:
                X_train, X_test, y_train, y_test = train_test_split(_X, _y,
                                                                    test_size=test_size,
                                                                    random_state=random_state)
            else:
                X_train = _X
                X_test = None
                y_train = _y
                y_test = None

            trains.update({cat: (X_train, y_train)})
            tests.update({cat: (X_test, y_test)})
        self.trains = trains
        self.tests = tests
            
    def transform(self):
        return self._make_data()
            
    def make_train_test_data(self):
        X_transf, y_transf = self.transform()
        if self._test_size and self._test_size is not None:
            self._train_test_split(X_transf, y_transf, 
                                  test_size=self._test_size, 
                                  random_state=self._random_state)
        else:
            self._train_test_split(X_transf, y_transf, 
                                  test_size=False,
                                  random_state=self._random_state)

    def feed_cat_Xy(self, cat:str, train:bool=True):
        if train:
            X, y = self.trains[cat]
        else:
            X, y = self.tests[cat]
        return X, y
        
    @property
    def main_keys(self) -> list:
        return self._main_keys
    
    @property
    def support_keys(self) -> list:
        return self._support_keys
