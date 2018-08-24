from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import os
import warnings
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        
        logger.info('\nTimings: %r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed

class ColumnSelector(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column selection
    
    Allows to select columns by name from pandas dataframes in scikit-learn
    pipelines.
    
    Parameters
    ----------
    columns : list of str, names of the dataframe columns to select
        Default: [] 
    
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """
        return self

    def transform(self, X):
        """ Selects columns of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        X : pandas DataFrame
            contains selected columns of X      
        """
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
            
            
class TypeSelector(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column selection by dtype
    
    Allows to select columns by dtype from pandas dataframes in scikit-learn
    pipelines.
    
    Parameters
    ----------
    dtype : dtype of the dataframe columns to select
            Default: [] 
    
    """
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        """ Do nothing operation
        
        Returns
        ----------
        self : object
        """
        return self

    def transform(self, X):
        """ Selects columns of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        X : pandas DataFrame
            contains selected columns of given dtype of X      
        """
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])
    
class ToDummiesTransformer(BaseEstimator, TransformerMixin):
    """ A Dataframe transformer that provide dummy variable encoding
    """
    
    def transform(self, X, **transformparams):
        """ Returns a dummy variable encoded version of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
        
        Returns
        ----------
        trans : pandas DataFrame
        
        """
    
        trans = pd.get_dummies(X).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        """ Do nothing operation
        
        Returns
        ----------
        self : object
        """
        return self