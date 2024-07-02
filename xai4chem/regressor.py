import joblib
import os
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import json
import optuna
import xgboost
import lightgbm
from catboost import CatBoostRegressor 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from featurewiz import FeatureWiz
from flaml.default import LGBMRegressor, XGBRegressor
from flaml.default import preprocess_and_suggest_hyperparams
from explain_model import explain_model


class Regressor:
    def __init__(self, output_folder,  algorithm='xgboost', n_trials=500, k=None):
        self.algorithm = algorithm
        self.n_trials = n_trials
        self.output_folder = output_folder
        self.model = None 
        self.max_features = k
        self.selected_features = None
        
    def _select_features(self, X_train, y_train):
        if self.max_features is None:
            print("No maximum feature limit specified. Using all features.")
            self.selected_features = list(X_train.columns)
        elif X_train.shape[1] <= self.max_features:
            print(f"Number of input features is less than or equal to {self.max_features}. Using all features.")
            self.selected_features = list(X_train.columns)
        else:
            print(f"Features in the dataset are more than {self.max_features}. Using Featurewiz for feature selection")
            fwiz = FeatureWiz(corr_limit=0.9, feature_engg='', category_encoders='', dask_xgboost_flag=False, nrows=None, verbose=0) 
            X_train_fwiz, _ = fwiz.fit_transform(X_train, y_train) 
            selected_features = fwiz.features
            
            if len(selected_features) >= self.max_features:
                print(f"Number of features selected by Featurewiz exceeds {self.max_features}. Selecting top {self.max_features}")
                self.selected_features = selected_features[:self.max_features]
            else: 
                print('Using Featurewiz,  skipping SULO algorithm in feature selection')
                fwiz = FeatureWiz(corr_limit=0.9, skip_sulov=True, feature_engg='', category_encoders='', dask_xgboost_flag=False, nrows=None, verbose=0) 
                X_train_fwiz, _ = fwiz.fit_transform(X_train, y_train) 
                selected_features = fwiz.features
                if len(selected_features) >= self.max_features:
                    print(f"Number of features selected by Featurewiz exceeds {self.max_features}. Selecting top {self.max_features}")
                    self.selected_features = selected_features[:self.max_features]
                else:
                    print(f"Number of features selected by Featurewiz is less than {self.max_features}. Using KBest selection.")
                    selector = SelectKBest(score_func=mutual_info_regression, k=self.max_features)
                    selector.fit(X_train, y_train)
                    # Get the indices of the selected features
                    selected_indices = np.argsort(selector.scores_)[::-1][:self.max_features]
                    self.selected_features = X_train.columns[selected_indices]
        
        return self.selected_features

    def _optimize_xgboost(self, trial, X, y):
        params = {  
            'lambda': trial.suggest_int('lambda', 0, 5),
            'alpha': trial.suggest_int('alpha', 0, 5), 
            'gamma': trial.suggest_int('gamma', 0,20),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001,1),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree',[0.4,0.5,0.6,0.7,0.8,1.0]),
            'colsample_bynode': trial.suggest_categorical('colsample_bynode',[0.4,0.5,0.6,0.7,0.8,1.0]),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'min_child_weight': trial.suggest_int('min_child_weight',1, 100),  
            'max_depth': trial.suggest_int('max_depth', 3, 20),  
            'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            'random_state': trial.suggest_categorical('random_state', [0, 42]),
            'early_stopping_rounds': 10
        }

        model = xgboost.XGBRegressor(**params)
        return self._train_and_evaluate_optuna_model(model, X, y)

    def _optimize_catboost(self, trial, X, y):
        params = {
            'iterations':trial.suggest_int("iterations", 500, 2000),
            'learning_rate' : trial.suggest_float('learning_rate',0.0001,1),
            'reg_lambda': trial.suggest_float('reg_lambda',1, 10),
            'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            'random_strength': trial.suggest_float('random_strength',1,10),
            'depth': trial.suggest_int('depth',5, 10),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',5,10),
            'early_stopping_rounds': 10
        }

        model = CatBoostRegressor(loss_function="RMSE", random_state=42, **params)
        return self._train_and_evaluate_optuna_model(model, X, y)

    def _train_and_evaluate_optuna_model(self, model, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42) 

        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        
        preds = model.predict(X_valid)
        mae = metrics.mean_absolute_error(y_valid, preds)
        return mae

    def fit(self, X_train, y_train, default_params=True):
        self._select_features(X_train, y_train)
        # print(f'Selected features: {self.selected_features}')
        X_train = X_train[self.selected_features]
        if self.algorithm == 'xgboost':
            if not default_params:
                study = optuna.create_study(direction="minimize")
                study.optimize(lambda trial: self._optimize_xgboost(trial, X_train, y_train), n_trials=self.n_trials)
                best_params = study.best_params
                print('Best parameters for XGBoost:', best_params)
                self.model = xgboost.XGBRegressor(**best_params)
            else:
                estimator = XGBRegressor()
                (
                    hyperparams,
                    estimator_name,
                    X_transformed,
                    y_transformed,
                ) = estimator.suggest_hyperparams(X_train, y_train)
                self.model = xgboost.XGBRegressor(**hyperparams)
            self.model.fit(X_train, y_train)                
        elif self.algorithm == 'catboost': 
            if not default_params:
                study = optuna.create_study(direction="minimize") 
                study.optimize(lambda trial: self._optimize_catboost(trial, X_train, y_train), n_trials=self.n_trials)
                best_params = study.best_params
                print('Best parameters for CatBoost:', best_params)
                self.model = CatBoostRegressor(**best_params)
            else:
                self.model = CatBoostRegressor()
            self.model.fit(X_train, y_train, verbose=False)
        elif self.algorithm == 'lgbm':
            estimator = LGBMRegressor()
            (
                hyperparams,
                estimator_name,
                X_transformed,
                y_transformed,
            ) = estimator.suggest_hyperparams(X_train, y_train)
                
            self.model = lightgbm.LGBMRegressor(**hyperparams)
            self.model.fit(X_train, y_train)          
        else:
            raise ValueError("Invalid Algorithm. Supported Algorithms: xgboost, catboost")


    def evaluate(self, X_valid_features, smiles_valid, y_valid):
        if self.model is None:
            raise ValueError("The model has not been trained.")

        y_pred = self.predict(X_valid_features)
        # A dataFrame with the predictions
        evaluation_data = pd.DataFrame({
            'SMILES': smiles_valid,
            'Actual Value': y_valid,
            'Predicted Value': y_pred
        })

        # Save as a CSV file
        evaluation_data.to_csv(os.path.join(self.output_folder, "evaluation_data.csv"), index=False)
        
        plt.scatter(y_valid, y_pred) 
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True vs. Predicted Values')
        plt.savefig(os.path.join(self.output_folder, 'evaluation_scatter_plot.png'))
        plt.close()
        
        # Metrics
        evaluation_metrics = {
            "Mean Squared Error": round(metrics.mean_squared_error(y_valid, y_pred), 4),
            "Root Mean Squared Error": round(np.sqrt(metrics.mean_squared_error(y_valid, y_pred)), 4),
            "Mean Absolute Error": round(metrics.mean_absolute_error(y_valid, y_pred), 4),
            "R-squared Score": round(metrics.r2_score(y_valid, y_pred), 4),
            "Explained Variance Score": round(metrics.explained_variance_score(y_valid, y_pred), 4)
            } 
        with open(os.path.join(self.output_folder, 'evaluation_metrics.json'), 'w') as f:
            json.dump(evaluation_metrics, f, indent=4)
            
        return evaluation_metrics
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("The model has not been trained.") 
        X = X[self.selected_features]  
        return self.model.predict(X)

    def explain(self, X_features, smiles_list=None, use_fingerprints=False):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        X = X_features[self.selected_features]
        explanation = explain_model(self.model,  X, smiles_list, use_fingerprints, self.output_folder)
        return explanation


    def save_model(self, filename):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        self.model = joblib.load(filename)
        