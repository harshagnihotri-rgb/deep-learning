from pathlib import Path
import json
from functools import lru_cache
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parent
NUMERIC = ['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','SpecialDay']
CATEGORICAL = ['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend']
DEFAULT_SESSION = {'Administrative':2,'Administrative_Duration':60.0,'Informational':0,'Informational_Duration':0.0,'ProductRelated':25,'ProductRelated_Duration':900.0,'BounceRates':0.01,'ExitRates':0.03,'SpecialDay':0.0,'Month':'Nov','OperatingSystems':2,'Browser':2,'Region':1,'TrafficType':2,'VisitorType':'Returning_Visitor','Weekend':False}

def load_metrics():
    return json.loads((ROOT/'models/metrics.json').read_text())

@lru_cache(maxsize=1)
def purchase_model():
    return joblib.load(ROOT/'models/purchase.joblib')

def predict_purchase(session):
    row = {**DEFAULT_SESSION, **session}
    for key in NUMERIC:
        value = float(row[key])
        if not 0 <= value < float('inf'):
            raise ValueError(f'{key} must be finite and non-negative')
        row[key] = value
    for key in ['BounceRates','ExitRates','SpecialDay']:
        if row[key] > 1: raise ValueError(f'{key} must be between 0 and 1')
    frame = pd.DataFrame([row], columns=NUMERIC+CATEGORICAL)
    p = float(purchase_model().predict_proba(frame)[0,1])
    return {'probability':p,'prediction':bool(p >= 0.5)}

def get_catalog():
    return pd.read_csv(ROOT/'data/catalog.csv')

@lru_cache(maxsize=1)
def catalog_index():
    catalog = get_catalog()
    matrix = TfidfVectorizer(stop_words='english', ngram_range=(1,2)).fit_transform(catalog['name']+' '+catalog['category']+' '+catalog['description'])
    return catalog, matrix

def recommend(product_id, top_n=4):
    catalog, matrix = catalog_index()
    found = catalog.index[catalog.product_id == product_id].tolist()
    if not found: raise ValueError('Unknown product ID')
    idx = found[0]
    scores = cosine_similarity(matrix[idx],matrix).ravel()
    indices = [i for i in scores.argsort()[::-1] if i != idx and scores[i] > 0][:max(0,int(top_n))]
    result = catalog.iloc[indices].copy()
    result['similarity'] = scores[indices]
    return result

@lru_cache(maxsize=1)
def sentiment_model(): return SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    if not isinstance(text,str) or not text.strip(): raise ValueError('Enter a non-empty review')
    scores = sentiment_model().polarity_scores(text)
    scores['label'] = 'Positive' if scores['compound'] >= .05 else 'Negative' if scores['compound'] <= -.05 else 'Neutral'
    return scores

@lru_cache(maxsize=1)
def image_model():
    import torch
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=None)
    path = ROOT/'models/mobilenet_v3_small.pth'
    if path.exists(): model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    else: model.load_state_dict(weights.get_state_dict(progress=False,check_hash=True))
    model.eval()
    return model, weights.transforms(), weights.meta['categories']

def classify_image(image):
    import torch
    from PIL import ImageOps
    model, transform, labels = image_model()
    image = ImageOps.exif_transpose(image).convert('RGB')
    with torch.inference_mode():
        scores = model(transform(image).unsqueeze(0)).softmax(dim=1)[0]
        values, indices = scores.topk(5)
    return [{'label':labels[int(i)],'probability':float(v)} for v,i in zip(values,indices)]
