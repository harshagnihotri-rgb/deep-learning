from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from xml.sax.saxutils import escape
import json
root=Path(__file__).resolve().parents[1]
sections={
'Project fundamentals':[
('What is RetailLens?', 'An e-commerce intelligence dashboard combining purchase-intent prediction, product similarity recommendations, image recognition, and review sentiment.'),
('What problem does it solve?', 'It turns browsing, product, image and review information into interpretable outputs for a retail demonstration. It does not implement payments or order management.'),
('What are the inputs and outputs?', 'Session features produce purchase probability; a product ID produces similar items; an uploaded image produces ranked ImageNet labels; an English review produces sentiment scores and a label.'),
('Is the entire project deep learning?', 'No. The image module uses a deep CNN. Random Forest is conventional machine learning; TF-IDF is statistical text representation; VADER is lexicon and rule based NLP.'),
('How does it differ from the proposal?', 'It uses real UCI sessions, saves reproducible evaluation, uses pretrained MobileNet instead of random-pixel CNN training, and handles sentiment with VADER rather than a tiny keyword list.'),
('What is your contribution?', 'Application integration, preprocessing and training pipeline, evaluation, input validation, recommendation pipeline, deployment configuration and documentation. Pretrained weights, datasets and libraries are credited to their authors.'),
('What is the architecture?', 'A Streamlit presentation layer calls engine.py. That layer loads a persisted purchase pipeline, builds a TF-IDF index, loads a pretrained CNN and runs VADER. Training is separate from serving.'),
('Why Python and Streamlit?', 'Python has mature ML libraries. Streamlit provides forms, uploads and tables with little infrastructure, making a reproducible academic demo practical.')],
'Purchase model and data':[
('Which dataset is used?', 'The UCI Online Shoppers Purchasing Intention dataset. Revenue is the target. Consult data/SOURCES.md and metrics.json for the exact source and evaluation details.'),
('What does a training row represent?', 'A browsing session described by page counts, durations, bounce/exit measures, calendar information and visitor/context categories.'),
('Why not keep the synthetic dataset?', 'A model trained on labels generated from a hand-written formula mainly learns that formula. It cannot establish real customer prediction quality.'),
('What is Random Forest?', 'An ensemble of decision trees using randomized samples and feature selection. Combining trees reduces variance relative to a single deep tree.'),
('How is prediction probability produced?', 'The classifier aggregates tree probability estimates. The result is an estimate and must be checked for calibration before interpreting it as an exact frequency.'),
('Why exclude PageValues?', 'It can encode transaction-related information inappropriate to a pre-purchase use case. Exclusion reduces one leakage risk; completed-session features still limit early-session claims.'),
('What is data leakage?', 'Information unavailable at prediction time, or information from the held-out test data, improperly influences training or model selection.'),
('Why use a preprocessing pipeline?', 'It keeps training and inference transformations consistent and fits preprocessing only on the training split.'),
('How are categorical features represented?', 'One-hot encoding creates indicator columns for categories. Unknown categories should be handled without crashing inference.'),
('Do trees require feature scaling?', 'Usually no: tree splits depend on ordering and thresholds, not distance between standardized features.'),
('What is a stratified split?', 'A split that approximately preserves class proportions in train and test sets, useful when purchases are less common.'),
('What is class imbalance?', 'One class appears much more often. A model predicting no purchase for everyone can have deceptively high accuracy.'),
('Why use a baseline?', 'It establishes whether the model improves on a simple strategy such as always predicting the majority class.'),
('Can this model predict actual live shoppers?', 'It can produce estimates, but the historical dataset is not proof of performance for a different shop or early session. Validate on recent, appropriately timed store data first.'),
('What is concept drift?', 'The relationship between inputs and outcomes changes over time, for example after pricing, traffic or seasonal changes.'),
('How would you improve the model?', 'Use timestamp-aware data collection, validation on later periods, better calibration, carefully tuned hyperparameters and store-specific features available before the prediction.')],
'Evaluation and statistics':[
('Define accuracy.', 'Correct predictions divided by all predictions. It is insufficient alone on imbalanced data.'),
('Define precision.', 'TP / (TP + FP): among predicted purchases, the fraction that really purchased.'),
('Define recall.', 'TP / (TP + FN): among actual purchases, the fraction detected.'),
('Define F1 score.', '2 * precision * recall / (precision + recall). It balances precision and recall but does not include true negatives.'),
('What is a confusion matrix?', 'A table of true negatives, false positives, false negatives and true positives at a chosen threshold. State the class ordering.'),
('What is ROC-AUC?', 'A threshold-independent ranking measure comparing true-positive and false-positive rates. It does not prove probability calibration.'),
('What is average precision?', 'A summary of precision across recall levels based on ranked predictions. It is useful alongside class prevalence for imbalanced tasks.'),
('What is the decision threshold?', 'The probability cutoff for a positive class. This demo uses 0.5; business costs may justify a different threshold chosen on validation data.'),
('What is overfitting?', 'Learning training-specific patterns that do not generalize. Control complexity and evaluate on unseen data.'),
('What is the difference between validation and test sets?', 'Validation guides choices; the final test estimates performance after those choices are fixed. Repeated tuning on the test set leaks information.'),
('Why fix a random seed?', 'It makes splitting and stochastic training more reproducible. Package versions and hardware can still affect results.'),
('What accuracy did you achieve?', 'Read the measured results section of this guide and models/metrics.json. Never quote the original PDF numbers as if they were reproduced.'),
('Does a high score prove real-world readiness?', 'No. Dataset representativeness, timing, leakage, calibration, subgroup performance and operational monitoring also matter.')],
'Product recommendations':[
('What kind of recommender is implemented?', 'Content-based item-to-item similarity using product names, categories and descriptions. It is not individual user personalization.'),
('What is TF-IDF?', 'Term frequency weights words in an item; inverse document frequency reduces the influence of words common across the catalog.'),
('What is cosine similarity?', 'The dot product of two vectors divided by the product of their lengths. It compares vector direction and helps rank similar text.'),
('Why remove stop words?', 'Very common words can add little distinguishing information. This choice is task dependent; sentiment should not blindly remove negation.'),
('What are n-grams?', 'Sequences of adjacent tokens. Bigrams can distinguish phrases such as gaming mouse from separate words.'),
('How is the queried product excluded?', 'The recommendation function filters out the query row before returning the highest scoring other products.'),
('What is cold start?', 'Insufficient interaction history for new users or products. Content-based methods can recommend new products when descriptions are available.'),
('How would you evaluate recommendations?', 'Use relevance judgments or held-out interactions for precision@k, recall@k or ranking metrics, then online experiments. This demo does not claim a measured recommendation accuracy.'),
('How does collaborative filtering differ?', 'It learns from user-item interactions rather than only item text. It needs sufficient interaction history and careful evaluation.')],
'CNN image module':[
('What is a CNN?', 'A neural network using learned local filters and hierarchical feature representations to analyze spatial inputs such as images.'),
('Why choose MobileNet V3 Small?', 'It is a compact pretrained CNN suitable for CPU inference in a demonstration, with lower compute needs than many large vision models.'),
('Did you train the CNN from scratch?', 'No. The application uses pretrained ImageNet weights. Fine-tuning on labeled store images is future work.'),
('Why was the original random-image training invalid?', 'Random pixels with random labels have no meaningful product-category relationship. Training accuracy on those samples does not establish product recognition.'),
('What image preprocessing is used?', 'The preprocessing bundled with the selected torchvision weights performs the expected resizing, cropping, tensor conversion and normalization. Training and serving preprocessing must match.'),
('What is convolution?', 'A small learned kernel slides across spatial positions to detect local patterns using shared weights.'),
('What is an activation function?', 'A nonlinear function that lets stacked layers represent nonlinear patterns. MobileNet uses architecture-specific activations rather than a single universal activation.'),
('What does pooling do?', 'It aggregates spatial responses, often reducing resolution and computation while retaining useful information.'),
('What is softmax?', 'It converts logits into nonnegative scores summing to one across the model categories. High softmax score is not a guarantee of correctness.'),
('What is transfer learning?', 'Reusing knowledge from a previously trained model. This app uses pretrained inference; fine-tuning would additionally update weights on target data.'),
('Can it return the four original product categories?', 'Not as a validated four-class classifier. It returns ImageNet labels. A trained and evaluated target-category head would be needed for the original four-category specification.'),
('Why use eval mode and inference mode?', 'Evaluation mode changes training-specific layer behavior; inference mode avoids gradient tracking, reducing memory and overhead.'),
('What happens with unfamiliar images?', 'The model still ranks its known categories and can be confidently wrong. Domain-specific testing and out-of-distribution handling are future improvements.')],
'Sentiment and language':[
('What is VADER?', 'Valence Aware Dictionary and sEntiment Reasoner: a lexicon and rule-based English sentiment method.'),
('What does compound mean?', 'A normalized combined sentiment score between -1 and 1. It is not a probability.'),
('How are sentiment labels assigned?', 'Compound at least 0.05 is Positive, at most -0.05 is Negative, otherwise Neutral.'),
('Why is not good a useful test?', 'It checks negation handling that simple positive-word matching often gets wrong.'),
('Why preserve punctuation and capitalization?', 'VADER can use them as intensity cues. Aggressive cleaning may remove useful sentiment information.'),
('What are its limitations?', 'Sarcasm, ambiguous context, domain-specific language, multilingual text and mixed opinions can produce incorrect labels.'),
('How could sentiment be improved?', 'Evaluate on labeled product reviews, compare with a fine-tuned transformer and inspect errors by language and product category.')],
'Deployment, security and viva demo':[
('Why save the trained model?', 'Serving should load a fixed trained artifact instead of retraining for every user interaction. This reduces latency and improves reproducibility.'),
('What is caching used for?', 'It reuses loaded models and the catalog index across repeated calls, avoiding expensive repeated initialization.'),
('What does GitHub deployment mean here?', 'GitHub stores versioned source. A Python-capable host such as Streamlit Community Cloud executes the dashboard. GitHub Pages alone cannot run it.'),
('What files are needed for hosting?', 'app.py, engine.py, requirements.txt, the product catalog and purchase model artifacts. CNN weights can be downloaded on first use if not bundled.'),
('How are secrets managed?', 'Never commit tokens or passwords. Use hosting secrets or environment configuration. This project does not require a paid API key for local inference.'),
('What upload checks are present?', 'The UI limits size and accepted formats, validates that Pillow can decode the image, and handles invalid images. These are basic protections, not a full production upload service.'),
('Why avoid untrusted model files?', 'Python pickle/joblib deserialization can execute code. Load only trusted artifacts generated or reviewed by the project.'),
('How would you scale the project?', 'Separate UI and inference API, add request limits and monitoring, use appropriate workers and model caching, and size compute after measurement.'),
('What is the main ethical concern?', 'Avoid overstating accuracy or using browsing predictions as certainty about individuals. Production tracking needs appropriate consent, access control and retention policies.'),
('How should you demonstrate the project?', 'Show architecture and real metrics, submit a session, find similar products, classify a photo, compare good with not good, and clearly state limitations.'),
('What if the internet fails during viva?', 'Prepare all dependencies, the saved purchase model and pretrained image weights beforehand. Local inference can then run without a hosted service.'),
('What are the next improvements?', 'Fine-tune target-category vision, evaluate sentiment with human labels, gather real recommendation judgments, calibrate purchase scores and connect a properly consented live data pipeline.')]
}
styles=getSampleStyleSheet()
styles['Title'].textColor=colors.HexColor('#102b46')
styles['Heading1'].textColor=colors.HexColor('#087f8c')
styles['BodyText'].fontSize=10
styles['BodyText'].leading=14
story=[Paragraph('RetailLens',styles['Title']),Paragraph('E-commerce Intelligence Platform',styles['Heading1']),Paragraph('Viva preparation guide - questions with model answers',styles['Heading2']),Paragraph('Prepared for the implemented project. Learn the explanations and relate them to the code; do not claim work or metrics that were not performed.',styles['BodyText']),Spacer(1,16)]
count=sum(map(len,sections.values()))
story.append(Paragraph(f'{count} viva questions covering architecture, machine learning, deep learning, evaluation and deployment.',styles['BodyText']))
story.append(Spacer(1,14))
for title in sections: story.append(Paragraph(title,styles['BodyText']))
story.append(PageBreak())
n=0
for title,items in sections.items():
    story.append(Paragraph(title,styles['Heading1']))
    for q,a in items:
        n+=1
        story.append(KeepTogether([Paragraph(f'{n}. {escape(q)}',styles['Heading3']),Paragraph(escape(a),styles['BodyText']),Spacer(1,7)]))
    story.append(Spacer(1,18))
story.append(PageBreak())
story.append(Paragraph('Measured purchase evaluation',styles['Heading1']))
metrics=root/'models/metrics.json'
if metrics.exists():
    measured=json.loads(metrics.read_text())
    for key in ['dataset','raw_rows','duplicates_removed','train_rows','test_rows','positive_rate','seed','threshold','accuracy','precision','recall','f1','roc_auc','average_precision','brier_score','baseline_accuracy','confusion_matrix','limitations']:
        value=measured.get(key)
        if isinstance(value,float): value=round(value,4)
        story.append(Paragraph(escape(key.replace('_',' ').title()+': '+str(value)),styles['BodyText']))
        story.append(Spacer(1,8))
else: story.append(Paragraph('Run scripts/train.py and consult models/metrics.json for measured evaluation.',styles['BodyText']))
story.append(PageBreak())
story.append(Paragraph('Sources and further reading',styles['Heading1']))
for title,url in [('UCI Online Shoppers dataset','https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset'),('Torchvision pretrained models','https://docs.pytorch.org/vision/stable/models.html'),('VADER source and documentation','https://github.com/cjhutto/vaderSentiment'),('Streamlit deployment','https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy'),('Scikit-learn documentation','https://scikit-learn.org/stable/')]:
    story.append(Paragraph(f'{escape(title)}: <link href="{url}" color="#087f8c">{escape(url)}</link>',styles['BodyText']));story.append(Spacer(1,12))
def footer(c,d):
    c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#607488'));c.drawString(44,26,'RetailLens | Viva preparation | 14 November 2026');c.drawRightString(550,26,str(d.page))
SimpleDocTemplate(str(root/'docs/viva-guide.pdf'),rightMargin=44,leftMargin=44,topMargin=40,bottomMargin=44).build(story,onFirstPage=footer,onLaterPages=footer)
print('Generated',count,'questions')
