import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from config import DATA_DIR, DTYPES, GRID_PARAMS
from transformers import (CategoriesExtractor, CountryTransformer, GoalAdjustor,
                          TimeTransformer)


def load_dataset(x_path, y_path):
    x = pd.read_csv(os.sep.join([DATA_DIR, x_path]),
                    dtype=DTYPES,
                    index_col="id")
    y = pd.read_csv(os.sep.join([DATA_DIR, y_path]))

    return x, y


def build_model():
    cat_processor = Pipeline([("transformer", CategoriesExtractor()),
                              ("one_hot",
                               OneHotEncoder(sparse=False,
                                             handle_unknown="ignore"))])

    country_processor = Pipeline([("transformer", CountryTransformer()),
                                  ("one_hot",
                                   OneHotEncoder(sparse=False,
                                                 handle_unknown="ignore"))])

    # Add your code here to create the missing
    # ColumnTransformer and Pipeline
    col_transformer = ColumnTransformer([("cat_", cat_processor, ['category']),
                                         ("country_", country_processor, ["counrtry"]),
                                         ('goal', GoalAdjustor(), ["goal", "static_usd_rate"]),
                                         ("time", TimeTransformer(), ["deadline", ...])
    ])
    
    model = Pipeline([("preprocessor", preprocessor), ("model", DecisionTreeClassifier())])
    
    return model


def tune_model():
    X_train, y_train = load_dataset(X_TRAIN, Y_TRAIN)
    model = build_model()
    gs = GridSearchCV(model, GRID_PARAMS, cv=3, n_jobs=-1)
    print(gs.best_score_)
    print(gs.best_params_)
    # get best params and retrain the model directly perhaps?
    
    
    
def train_model(print_params=False):
    X_train, y_train = load_dataset(X_TRAIN, Y_TRAIN)
    model = build_model()
    
    model.fit(X_train, y_train)
    
    joblib.dump(model, "my_model.joblib")


def test_model():
    X_test, y_test = load_dataset(X_TEST, Y_TEST)
    model.joblib("my_model.joblib")
    y_pred - model.predict(X_TEST)
    print("Accuracy")
    print(classification_report(X_test, y_test))
    
    
    
    
