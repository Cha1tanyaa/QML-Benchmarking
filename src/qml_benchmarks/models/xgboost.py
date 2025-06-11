# Copyright 2025 Chaitanya Agrawal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

class XGBoost(BaseEstimator, ClassifierMixin): #XGBoost classifier
    def __init__(
        self,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        objective="binary:logistic",
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        scaling=1.0,
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.random_state = random_state
        self.scaling = scaling
        self.model = None
        self.scaler = None
        #self._label_mapping = None
        #self._reverse_label_mapping = None

    def fit(self, X, y):
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.transform(X)

        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("XGBoostClassifier supports binary classification.")
        if set(unique_classes) != {0, 1}:
            mapping = {unique_classes[0]: 0, unique_classes[1]: 1}
            y = np.array([mapping[val] for val in y])

        self.model = xgb.XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            objective=self.objective,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = self.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        X = self.transform(X)
        return self.model.predict_proba(X)

    def transform(self, X):
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(X)

        X = self.scaler.transform(X) * self.scaling
        return X
