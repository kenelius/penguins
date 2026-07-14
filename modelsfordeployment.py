import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# 1. Load Data
print("Loading data...")
data = pd.read_csv("IRIS.csv")
X = data.drop(columns=['species'])
y = data['species']

# 2. Preprocessing Setup
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, num_features)],
    remainder='drop'
)

# Split Data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# --- Model 1: Logistic Regression ---
print("Training Logistic Regression...")
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])
param_dist_lr = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear', 'saga'],
    'classifier__class_weight': [None, 'balanced']
}
search_lr = RandomizedSearchCV(lr_pipeline, param_dist_lr, n_iter=20, cv=5, scoring='f1_weighted', random_state=42, n_jobs=-1)
search_lr.fit(X_train, y_train)
joblib.dump(search_lr.best_estimator_, 'model_lr.pkl')

# --- Model 2: KNN ---
print("Training KNN...")
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('knn', KNeighborsClassifier())
])
param_dist_knn = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['minkowski', 'euclidean']
}
search_knn = RandomizedSearchCV(knn_pipeline, param_dist_knn, n_iter=20, cv=5, scoring='f1_weighted', random_state=42, n_jobs=-1)
search_knn.fit(X_train, y_train)
joblib.dump(search_knn.best_estimator_, 'model_knn.pkl')

# --- Model 3: Random Forest ---
print("Training Random Forest...")
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
])
param_dist_rf = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 10, 15],
    'rf__min_samples_split': [2, 5],
    'rf__class_weight': ['balanced']
}
search_rf = RandomizedSearchCV(rf_pipeline, param_dist_rf, n_iter=20, cv=5, scoring='f1_weighted', random_state=42, n_jobs=-1)
search_rf.fit(X_train, y_train)
joblib.dump(search_rf.best_estimator_, 'model_rf.pkl')

# --- Model 4: XGBoost ---
print("Training XGBoost...")
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(eval_metric='mlogloss', random_state=42))
])
param_dist_xgb = {
    'classifier__n_estimators': [100, 300],
    'classifier__max_depth': [3, 6, 10],
    'classifier__learning_rate': [0.01, 0.1, 0.3],
    'classifier__subsample': [0.6, 0.8, 1.0]
}
search_xgb = RandomizedSearchCV(xgb_pipeline, param_dist_xgb, n_iter=20, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
search_xgb.fit(X_train, y_train)
joblib.dump(search_xgb.best_estimator_, 'model_xgb.pkl')

# Save Label Encoder and Feature Names for the Dashboard
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(num_features, 'feature_names.pkl')

print("\n✅ All models and artifacts saved successfully!")
print("You can now run your Streamlit app with: streamlit run app.py")