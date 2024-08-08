import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import optuna
import xgboost
import lightgbm
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from flaml.default import LGBMClassifier, XGBClassifier
from featurewiz import FeatureWiz
from xai4chem.reporting import explain_model, classification_metrics


class Classifier:
    def __init__(self, output_folder,  fingerprints=None, algorithm='xgboost', n_trials=500, k=None):
        self.algorithm = algorithm
        self.n_trials = n_trials
        self.output_folder = output_folder
        self.model = None
        self.max_features = k
        self.selected_features = None
        self.fingerprints = fingerprints 
        self.optimal_threshold = None  # Store the optimal threshold

    def _select_features(self, X_train, y_train):
        if self.max_features is None:
            print("No maximum feature limit specified. Using all features.")
            self.selected_features = list(X_train.columns)
        elif X_train.shape[1] <= self.max_features:
            print(f"Number of input features is less than or equal to {self.max_features}. Using all features.")
            self.selected_features = list(X_train.columns)
        else:
            print(f"Features in the dataset are more than {self.max_features}. Using Featurewiz for feature selection")
            fwiz = FeatureWiz(corr_limit=0.9, feature_engg='', category_encoders='', dask_xgboost_flag=False,
                              nrows=None, verbose=0)
            X_train_fwiz, _ = fwiz.fit_transform(X_train, y_train)
            selected_features = fwiz.features

            if len(selected_features) >= self.max_features:
                print(
                    f"Selecting top {self.max_features}")
                self.selected_features = selected_features[:self.max_features]
            else:
                print('Using Featurewiz, skipping SULO algorithm in feature selection')
                fwiz = FeatureWiz(corr_limit=0.9, skip_sulov=True, feature_engg='', category_encoders='',
                                  dask_xgboost_flag=False, nrows=None, verbose=0)
                X_train_fwiz, _ = fwiz.fit_transform(X_train, y_train)
                selected_features = fwiz.features
                if len(selected_features) >= self.max_features:
                    print(
                        f"Selecting top {self.max_features}")
                    self.selected_features = selected_features[:self.max_features]
                else:
                    print(
                        f"Number of features selected by Featurewiz is less than {self.max_features}. Using KBest selection.")
                    selector = SelectKBest(score_func=mutual_info_classif, k=self.max_features)
                    selector.fit(X_train, y_train)
                    selected_indices = np.argsort(selector.scores_)[::-1][:self.max_features]
                    self.selected_features = X_train.columns[selected_indices]
        return self.selected_features

    def _optimize_xgboost(self, trial, X, y):
        params = {
            'lambda': trial.suggest_int('lambda', 0, 5),
            'alpha': trial.suggest_int('alpha', 0, 5),
            'gamma': trial.suggest_int('gamma', 0, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'colsample_bynode': trial.suggest_categorical('colsample_bynode', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'random_state': trial.suggest_categorical('random_state', [0, 42]),
            'early_stopping_rounds': 10
        }

        model = XGBClassifier(**params)
        return self._train_and_evaluate_optuna_model(model, X, y)

    def _optimize_catboost(self, trial, X, y):
        params = {
            'iterations': trial.suggest_int("iterations", 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'random_strength': trial.suggest_float('random_strength', 1, 10),
            'depth': trial.suggest_int('depth', 5, 10),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 5, 10),
            'early_stopping_rounds': 10
        }

        model = CatBoostClassifier(loss_function="Logloss", random_state=42, **params)
        return self._train_and_evaluate_optuna_model(model, X, y)

    def _train_and_evaluate_optuna_model(self, model, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        preds = model.predict(X_valid)
        accuracy = metrics.accuracy_score(y_valid, preds)
        return accuracy

    def fit(self, X_train, y_train, default_params=True):
        self._select_features(X_train, y_train)
        X_train = X_train[self.selected_features]

        if self.algorithm == 'xgboost':
            if not default_params:
                study = optuna.create_study(direction="maximize")
                study.optimize(lambda trial: self._optimize_xgboost(trial, X_train, y_train), n_trials=self.n_trials)
                best_params = study.best_params
                print('Best parameters for XGBoost:', best_params)
                self.model = xgboost.XGBClassifier(**best_params)
            else:
                estimator = XGBClassifier()
                (
                    hyperparams,
                    estimator_name,
                    X_transformed,
                    y_transformed,
                ) = estimator.suggest_hyperparams(X_train, y_train)

            self.model = xgboost.XGBClassifier(**hyperparams)
            self.model.fit(X_train, y_train)
        elif self.algorithm == 'catboost':
            if not default_params:
                study = optuna.create_study(direction="maximize")
                study.optimize(lambda trial: self._optimize_catboost(trial, X_train, y_train), n_trials=self.n_trials)
                best_params = study.best_params
                print('Best parameters for CatBoost:', best_params)
                self.model = CatBoostClassifier(**best_params)
            else:
                self.model = CatBoostClassifier()
            self.model.fit(X_train, y_train, verbose=False)
        elif self.algorithm == 'lgbm':
            estimator = LGBMClassifier()
            (
                hyperparams,
                estimator_name,
                X_transformed,
                y_transformed,
            ) = estimator.suggest_hyperparams(X_train, y_train)

            self.model = lightgbm.LGBMClassifier(**hyperparams)
            self.model.fit(X_train, y_train)
        else:
            raise ValueError("Invalid Algorithm. Supported Algorithms: xgboost, catboost")

    def evaluate(self, X_valid_features, smiles_valid, y_valid):
        if self.model is None:
            raise ValueError("The model has not been trained.")

        y_proba, _ = self.model_predict(X_valid_features)

        # Calculate ROC curve and optimal thresholds
        fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_proba)
        roc_auc = metrics.auc(fpr, tpr)

        self.optimal_threshold = thresholds[np.argmax(tpr - fpr)]  # Youden's J statistic

        eval_metrics = classification_metrics(smiles_valid, y_valid, y_proba, self.output_folder)

        return eval_metrics

    def model_predict(self, X):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        print('..predicting..')
        X = X[self.selected_features]
        y_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        # if self.optimal_threshold is not None:
        #     y_pred_optimal = (y_proba >= self.optimal_threshold).astype(int)
        #     return y_proba, y_pred, y_pred_optimal
        # else:
        return y_proba, y_pred

    def explain(self, X_features, smiles_list=None):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        X = X_features[self.selected_features]
        explanation = explain_model(self.model, X, smiles_list, self.output_folder, self.fingerprints)
        return explanation

    def save_model(self, filename):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        model_data = {
            'model': self.model,
            'selected_features': self.selected_features,
            'fingerprints': self.fingerprints, 
            'optimal_threshold': self.optimal_threshold
        }
        joblib.dump(model_data, filename)

    def load_model(self, filename):
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.selected_features = model_data['selected_features']     
        self.optimal_threshold = model_data['optimal_threshold']
        self.fingerprints = model_data["fingerprints"] 