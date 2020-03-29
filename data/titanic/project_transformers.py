"""
Customer tranformers for this specific project / data
"""

import re
import itertools
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class DataFrameSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.columns = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.columns = list(X.columns)
        return X

    def get_feature_names(self):
        return self.columns



class GetNameFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, as_string=True):
        self.as_string = as_string
        self.columns = None
        
    def fit(self, X, y=None):
        return self
    
    def __get_prefix(self, string):
        
        categories = {"Mr.": 1, "Mrs.": 2, "Miss.": 3, "Master.": 4}
        prefix = re.findall("Mr\.|Mrs\.|Miss\.|Master.", string)
        
        if prefix:
            return prefix[0] if self.as_string else categories[prefix[0]]

        else:
            return "None" if self.as_string else  0
        
        
    def __get_parenths(self, string):
        if len(re.findall(("\("), string)) > 0:
            return "yes" if self.as_string  else 1

        else:
            return "no" if self.as_string else 0

        
    def transform(self, X, y=None):
        
        df = X.copy(deep=True)
        
        self.columns = ['name_prefix', 'name_parenths']
        
        df[self.columns[0]] = X.applymap(self.__get_prefix)
        df[self.columns[1]] = X.applymap(self.__get_parenths)
        
        return df[self.columns]
        
    
    def get_feature_names(self):
        return self.columns


class TicketTextBin(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.columns = ["ticket_text", "ticket_length"]
        
        
    def fit(self, X, y=None):
        return self


    def __ticket_chars_bin(self, string):
    # Return a bin number based on the ticket characters

        cats = {"PC": "2", "CA": "3", "A5": "4", "STONO2": "5",        
                "SOTONOQ": "6", "SCPARIS": "7", "WC": "8"}

        catch_all_bin = len(cats.keys()) + 1

        if string == "None":
            return "1"
        else:
            return cats.get(string, catch_all_bin)
        

    def __ticket_chars_extract(self, string):
    # Return the cabin number characters if available
        string_search = re.search("(^[A-Z].*(?=(\s)))", str(string))
        
        if string_search:
            ticket_chars =  re.sub("[ ./]+", "", string_search.group(1).upper()) 
        else:
            ticket_chars = "None"
            
        #return self.ticket_chars_bin(ticket_chars)
        return self.__ticket_chars_bin(ticket_chars)
    
    
    def __ticket_length(self, string):
        if string == "None":
            return string.upper()
        else:
            string_search = re.search("([0-9]*$)", str(string))
            
        string_length = len(string_search.group(1))
        
        return "3" if string_length <= 3 else str(string_length)
            
    
    def transform(self, X, y=None):
        df = X.copy(deep=True)

        df[self.columns[0]] = X.applymap(self.__ticket_chars_extract)
        df[self.columns[1]] = X.applymap(self.__ticket_length)

        return df[self.columns]
    
    def get_feature_names(self):
        return self.columns
    
    
class CabinCharacters(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = ["cabin_chars"]
        
        
    def fit(self, X, y=None):
        return self
    

    def __bin_cabin_chars(self, string):
        infrequent = ['T', 'G', 'FG', 'FE']

        if string in infrequent:
            return "INFREQ"
        else:
            return string.upper()

    
    def __cabin_chars_extractor(self, string):
        "Extract starting cabin characters if available"

        if string == "None":
            return string

        string = re.sub("[\s]+", "", str(string))

        string_search = re.search("(^[A-Z]+)", string)

        if string_search:
            cabin_chars =  string_search.group(1)
        else:
            cabin_chars = "None"
            
        return self.__bin_cabin_chars(cabin_chars)
       
    
    def transform(self, X, y=None):
        df = pd.DataFrame()
        df[self.columns] = X.applymap(self.__cabin_chars_extractor)
        return df
        
    def get_feature_names(self):
        return self.columns


def get_feature_names(preprocessor_name):


    # The final transformer is a parameter of the ColumnTransformer; ignore it
    stages = [pipeline[1] for pipeline in preprocessor_name.transformers_[:-1]]
    pipelines = [stage.steps for stage in stages]

    # Get the last step as well, skip the string representation of the name
    # and return only the transformer class
    last_step = [transformer[-1][1] for transformer in pipelines]

    # This returns a list of lists
    var_names = [step.get_feature_names() for step in last_step]

    # Flatten into a single list
    var_names  = list(itertools.chain(*var_names))

    return var_names
