import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

print("1. Loading dataset.csv...")
df = pd.read_csv('dataset.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

category_mapping = {'0=Blood Donor': 0, '0s=suspect Blood Donor': 1, '1=Hepatitis': 1, '2=Fibrosis': 2, '3=Cirrhosis': 3}
df['Category'] = df['Category'].map(category_mapping)
df['Sex'] = df['Sex'].map({'m': 1, 'f': 0})

X = df.drop(columns=['Category'])
y = df['Category']

print("2. Preprocessing Data...")
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

print("3. Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("4. Training Stacking Ensemble... (This takes about 5-15 seconds)")
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ('xgb', XGBClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss', random_state=42, n_jobs=-1)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
]
meta_model = LogisticRegression(max_iter=1000, random_state=42)
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)

stacking_clf.fit(X_train_smote, y_train_smote)

print("5. Saving perfectly matched .pkl files locally...")
joblib.dump(stacking_clf, 'fld_stacking_model.pkl')
joblib.dump(scaler, 'fld_scaler.pkl')
joblib.dump(imputer, 'fld_imputer.pkl')

print("SUCCESS! Your files are now updated.")