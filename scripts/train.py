"""Reproducible holdout evaluation; no training occurs during app requests."""
import sys, json, hashlib, io, zipfile, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
from engine import NUMERIC,CATEGORICAL
URL='https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip'
ROOT.joinpath('data').mkdir(exist_ok=True)
ROOT.joinpath('models').mkdir(exist_ok=True)
path=ROOT/'data/online_shoppers_intention.csv'
if not path.exists():
    payload=urllib.request.urlopen(URL,timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        path.write_bytes(z.read('online_shoppers_intention.csv'))
raw=pd.read_csv(path)
df=raw.drop_duplicates().reset_index(drop=True)
X=df[NUMERIC+CATEGORICAL]; y=df.Revenue.astype(int)
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
preprocess=ColumnTransformer([('numbers','passthrough',NUMERIC),('categories',OneHotEncoder(handle_unknown='ignore'),CATEGORICAL)])
model=Pipeline([('preprocess',preprocess),('classifier',RandomForestClassifier(n_estimators=200,max_depth=14,min_samples_leaf=4,class_weight='balanced_subsample',random_state=42,n_jobs=-1))])
model.fit(Xtrain,ytrain)
p=model.predict_proba(Xtest)[:,1]; pred=p>=.5
baseline=DummyClassifier(strategy='most_frequent').fit(Xtrain,ytrain).predict(Xtest)
metrics={'dataset':'UCI Online Shoppers Purchasing Intention (468)','source':URL,'license':'CC BY 4.0','dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'raw_rows':len(raw),'duplicates_removed':len(raw)-len(df),'rows':len(df),'train_rows':len(Xtrain),'test_rows':len(Xtest),'positive_rate':float(y.mean()),'seed':42,'threshold':.5,'excluded_features':['PageValues'],'accuracy':accuracy_score(ytest,pred),'precision':precision_score(ytest,pred),'recall':recall_score(ytest,pred),'f1':f1_score(ytest,pred),'roc_auc':roc_auc_score(ytest,p),'average_precision':average_precision_score(ytest,p),'brier_score':brier_score_loss(ytest,p),'baseline_accuracy':accuracy_score(ytest,baseline),'confusion_matrix':confusion_matrix(ytest,pred).tolist(),'limitations':['Random stratified holdout, not a temporal or external validation.','Session-level retrospective prediction; not validated for early-session live conversion.','PageValues excluded to reduce target-leakage risk.','Probabilities are uncalibrated model scores.']}
joblib.dump(model,ROOT/'models/purchase.joblib',compress=3)
(ROOT/'models/metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))

