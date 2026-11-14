# RetailLens - E-commerce Intelligence

A runnable academic application with four modules: purchase intent, content-based product recommendations, pretrained CNN image classification, and English review sentiment.

## Run locally (Python 3.11)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

If model artifacts are missing, run `python scripts/train.py` using the activated environment. Internet is needed for the UCI training dataset and first CNN weight download. The dashboard itself uses local inference. Do not load untrusted joblib model files.

## Architecture

Streamlit forms call `engine.py`. The purchase pipeline preprocesses UCI browsing sessions and applies a Random Forest. A sparse TF-IDF index ranks a curated demonstration catalog by cosine similarity. MobileNet V3 Small classifies uploaded images using ImageNet labels. VADER supplies lexicon/rule-based review sentiment. Models are cached across calls.

## Scope and honest limitations

- The purchase dataset is historical UCI data, not live telemetry from a store. PageValues is excluded to reduce leakage risk. Completed-session features do not establish early-session performance.
- Evaluation numbers are recorded in `models/metrics.json`; they must not be replaced with the unverified numbers in the original proposal.
- Recommendations are item similarity, not learned user personalization. Catalog names, descriptions and prices are demo content.
- The CNN is pretrained, not trained from scratch by this project. Its ImageNet labels replace the proposal's unsupported four-class random-pixel demonstration. Fine-tuning four store categories needs a labeled dataset and independent test set.
- Sentiment is English rule-based NLP, not a neural network. Sarcasm and mixed opinions can fail.
- No checkout, payments, accounts, or persistent customer tracking are implemented.

## Reproduce and test

```powershell
python scripts/train.py
python -m pytest -q
```

See `data/SOURCES.md` for attribution and `docs/viva-guide.pdf` for viva preparation. Training uses a fixed seed and separates held-out evaluation from model fitting. A majority baseline is essential because purchases are a minority class.

## Publish source to GitHub

Create a repository, then push this project directory. Keep `.env`, private keys and `.streamlit/secrets.toml` out of Git. The supplied `.gitignore` excludes those files and temporary files. The attributed UCI CSV is included for reproducibility. GitHub stores the source; GitHub Pages cannot execute this Python application.

## Make the app live

1. Push this directory to your GitHub repository, including required model artifacts and `data/catalog.csv`.
2. Open https://share.streamlit.io and connect GitHub.
3. Create an app, choose this repository, branch and `app.py`.
4. Use Python 3.11 in advanced settings and deploy.
5. Test each tab through the generated public URL. On a resource-limited host, CNN initialization may be slow; choose sufficient memory if needed.

Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Suggested viva demonstration

Explain the four modules, show held-out metrics and the baseline, submit a purchase session, select headphones for similar products, upload an object photograph, then compare 'good' with 'not good'. End by explaining dataset limitations and the difference between pretrained inference and fine-tuning.

## Verified results

On 2,441 held-out sessions after duplicate removal: ROC-AUC 0.7826, purchase precision 0.3344, recall 0.5497, F1 0.4158. Accuracy is 0.7583, below the majority baseline of 0.8435. The balanced model detects purchases at the cost of false positives; it must not be advertised as a production-ready high-accuracy system. No independent product-image or sentiment benchmark is claimed.

The pretrained CNN was smoke-tested on the official PyTorch example image. The local Streamlit health endpoint returned `ok`. Browser visual automation was unavailable, so visual browser QA remains unverified. The PDF was rendered and visually reviewed.
