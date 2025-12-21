from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


def preprocess_data(X: pd.DataFrame):
    # Categorical and numerical columns
    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(include="number").columns
    # Create the pipeline
    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
    )
    pre = ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
    ).set_output(transform="pandas")
    return pre


def get_models():
    RANDOM_STATE = 42
    models = [
        LogisticRegression(random_state=RANDOM_STATE),
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        RandomForestClassifier(random_state=RANDOM_STATE, max_depth=3),
        HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_depth=3),
        XGBClassifier(n_estimators=200, max_depth=3),
    ]
    return models


def evaluate_single_model(model, xtrain, ytrain, xtest, ytest):
    cv_scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring="f1_macro")
    model.fit(xtrain, ytrain)    
    ypred_train = model.predict(xtrain)
    f1_train = f1_score(ytrain, ypred_train, average="macro")
    ypred_test = model.predict(xtest)
    f1_test = f1_score(ytest, ypred_test, average="macro")
    res = {
        "model": model,
        "name": type(model).__name__,
        "f1_train": round(f1_train, 4),
        "f1_test": round(f1_test,4),
        "f1_cv": cv_scores.mean().round(4)
    }
    return res


def algo_evaluation(models: list, xtrain, ytrain, xtest, ytest):
    results = []
    for i in models:
        r = evaluate_single_model(i, xtrain, ytrain, xtest, ytest)
        print(r)
        results.append(r)
        print("="*100)
    res_df = pd.DataFrame(results)
    sort_df = (
        res_df.sort_values(by = "f1_cv", ascending=False)
        .reset_index(drop = True)
    )
    best_model = sort_df.loc[0, "model"]
    return sort_df, best_model
