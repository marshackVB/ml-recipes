
import pickle

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

from transformers.general import CategoricalEncoder
from project_transformers import (DataFrameSelector, GetNameFeatures, TicketTextBin, 
                                  CabinCharacters, get_feature_names)



def ingest_data():
    """Ingest data from csv files, apply schema, an union
    """

    types = {'PassengerId': str, 'Survived': int, 'Pclass': str, 'Name': str, 
             'Sex': str, 'Age': float, 'SipSp': int, 'Parch': int,'Ticket': str, 
             'Fare': float, 'Cabin': str, 'Embarked': str}

    titanic_raw = pd.read_csv("train.csv", dtype=types)

    # Age has missing values, fill with the mean age for now
    mean_age = titanic_raw['Age'].mean()
    titanic_raw['Age'] = titanic_raw['Age'].fillna(mean_age)

    return titanic_raw


def create_features():
    """Generate features
    """

    titanic_raw = ingest_data()


    # Make pipelines
    name_extractor = make_pipeline(GetNameFeatures(as_string=True),
                                   CategoricalEncoder())

    categoricals = make_pipeline(CategoricalEncoder())

    ticket_text = make_pipeline(TicketTextBin(),
                                CategoricalEncoder())

    cabin_chars = make_pipeline(CabinCharacters(),
                                CategoricalEncoder())

    passthrough = make_pipeline(DataFrameSelector())


    # Connect pipelines
    preprocessor = ColumnTransformer(transformers=[

                        ('name_extractor', name_extractor, ["Name"]),
                        ('categoricals',   categoricals,   ["Sex", "Embarked", "Pclass"]),
                        ('ticket_text',    ticket_text,    ["Ticket"]),
                        ('cabin_chars',    cabin_chars,    ["Cabin"]),
                        ('passthrough',    passthrough,    ["Age", "Fare", "SibSp", "Parch", "Survived"])
                            ]
                        )


    features = preprocessor.fit_transform(titanic_raw)

    feature_names = get_feature_names(preprocessor)

    # Create a DataFrame from the numpy features and feature names 
    features_df = pd.DataFrame(features, columns=feature_names)

    label_col = "Survived"
    feature_cols = [col for col in features_df.columns if col != label_col]
    # Excluding the label col
    categorical_cols = ['name_prefix',
                        'name_parenths',
                        'Sex',
                        'Embarked',
                        'Pclass',
                        'ticket_text',
                        'ticket_length',
                        'cabin_chars',
                        'SibSp',
                        'Parch']

    return (features_df, label_col, feature_cols, categorical_cols)


if __name__ == "__main__":

    features_df = create_features()

    pickle.dump(features_df, open("features_df_categorical.p", "wb"))





