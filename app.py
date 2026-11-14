"""Retail intelligence dashboard. Run: streamlit run app.py."""
import streamlit as st
from PIL import Image, UnidentifiedImageError
import engine

st.set_page_config(page_title="RetailLens | Commerce Intelligence", page_icon="???", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"],.stApp{font-family:Inter,sans-serif}
.stApp{background:#f4f7fb} h1,h2,h3{color:#153451}
.block-container{max-width:1250px;padding-top:2.5rem}
[data-testid="stSidebar"]{background:#e7eef6}
[data-testid="stMetric"]{background:white;border:1px solid #dbe5ef;padding:20px;border-radius:16px}
[data-testid="stForm"]{background:white;border:1px solid #dbe5ef;border-radius:16px;padding:24px}
.stButton button,.stFormSubmitButton button{border-radius:10px}
button[data-baseweb="tab"]{font-size:15px;padding:14px}
</style>""", unsafe_allow_html=True)
st.caption("RETAILLENS ? E-COMMERCE INTELLIGENCE")
st.title("Understand your customers. Discover your products.")
st.write("Choose a tool below to explore a shopping session, find similar products, identify an image, or analyze a review.")
with st.sidebar:
    st.header("RetailLens")
    st.write("Your retail intelligence workspace")
    st.divider()
    st.markdown("**Quick demo**")
    st.write("Predict a session ? Find similar products ? Upload an image ? Analyze a review")
    st.divider()
    st.caption("Built with Python, Random Forest, TF-IDF, MobileNet V3 and VADER.")
    st.info("Academic demonstration. Model estimates need validation on your own store.")
tabs = st.tabs(["Overview", "Purchase intent", "Similar products", "Image lab", "Review sentiment"])
with tabs[0]:
    st.subheader("What would you like to explore?")
    for col, title, body in zip(st.columns(4), ["Predict", "Discover", "Classify", "Listen"], ["Estimate purchase intent from browsing sessions.", "Find products with similar descriptions.", "Inspect pretrained CNN image predictions.", "Score positive and negative customer language."]):
        col.markdown(f"### {title}"); col.write(body)
    st.subheader("Measured evaluation")
    try:
        metrics = engine.load_metrics()
        a, b, c, d = st.columns(4)
        a.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        b.metric("Purchase recall", f"{metrics['recall']:.1%}")
        c.metric("Purchase precision", f"{metrics['precision']:.1%}")
        d.metric("Held-out sessions", f"{metrics['test_rows']:,}")
        st.caption(f"Accuracy: {metrics['accuracy']:.1%} · Majority baseline: {metrics['baseline_accuracy']:.1%} · F1: {metrics['f1']:.3f}")
        with st.expander("Evaluation details and limitations"):
            st.dataframe({"Measure": ["Training sessions", "Test sessions", "Accuracy", "Majority baseline accuracy", "F1 score"], "Value": [str(metrics["train_rows"]), str(metrics["test_rows"]), f"{metrics['accuracy']:.1%}", f"{metrics['baseline_accuracy']:.1%}", f"{metrics['f1']:.3f}"]}, hide_index=True)
            st.write("ROC-AUC measures how well the model ranks purchasing sessions. Recall measures how many actual purchases it detects. Precision measures how many predicted purchases are correct.")
            st.caption("Historical session data, random holdout evaluation, and uncalibrated probability estimates.")
    except (FileNotFoundError, RuntimeError) as exc:
        st.warning(f"Train the purchase model first: python scripts/train.py. {exc}")
    st.caption("Image recognition uses pretrained ImageNet categories. Recommendations use similarities between product descriptions.")
with tabs[1]:
    st.subheader("Will this browsing session convert?")
    st.write("Uses session browsing summaries; live early-session predictions are not validated. PageValues is excluded to reduce leakage risk.")
    with st.form("purchase"):
        left, right = st.columns(2)
        with left:
            admin = st.number_input("Account / administrative pages viewed", 0, 100, 1, help="Number of administrative pages visited during the session.")
            admin_time = st.number_input("Administrative duration (seconds)", 0.0, 100000.0, 30.0)
            info = st.number_input("Information pages viewed", 0, 100, 0, help="Pages containing information about the website or store.")
            info_time = st.number_input("Informational duration (seconds)", 0.0, 100000.0, 0.0)
            product = st.number_input("Product pages", 0, 1000, 15)
            product_time = st.number_input("Product duration (seconds)", 0.0, 100000.0, 500.0)
            bounce = st.slider("Bounce rate", 0.0, 1.0, 0.02, help="0.02 means 2%. Dataset measure for visited pages.")
            exits = st.slider("Exit rate", 0.0, 1.0, 0.04, help="0.04 means 4%. Dataset measure of exits from visited pages.")
        with right:
            special = st.slider("Closeness to a special day", 0.0, 1.0, 0.0)
            month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
            visitor = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"], format_func=lambda value: value.replace("_", " "))
            weekend = st.checkbox("Weekend")
        with st.expander("Advanced session details (defaults are ready for a demo)"):
            st.caption("These are anonymous category codes from the training dataset.")
            os_id = st.number_input("Operating system code", 1, 8, 2)
            browser = st.number_input("Browser code", 1, 13, 2)
            region = st.number_input("Region code", 1, 9, 1)
            traffic = st.number_input("Traffic source code", 1, 20, 2)
        submit = st.form_submit_button("Estimate purchase intent", type="primary")
    if submit:
        row = dict(Administrative=admin, Administrative_Duration=admin_time, Informational=info, Informational_Duration=info_time, ProductRelated=product, ProductRelated_Duration=product_time, BounceRates=bounce, ExitRates=exits, SpecialDay=special, Month=month, VisitorType=visitor, Weekend=weekend, OperatingSystems=os_id, Browser=browser, Region=region, TrafficType=traffic)
        try:
            result = engine.predict_purchase(row)
            st.metric("Estimated purchase probability", f"{result['probability']:.1%}")
            st.progress(result["probability"])
            st.write("The model predicts a purchase at the 50% threshold." if result["prediction"] else "The estimate falls below the 50% purchase threshold.")
            st.caption("Model probability is an estimate; validation on your store is required before business use.")
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
with tabs[2]:
    st.subheader("Find the next relevant product")
    catalog = engine.get_catalog()
    names = catalog.set_index("product_id")["name"].to_dict()
    selected = st.selectbox("Choose a product", catalog["product_id"].tolist(), format_func=lambda product_id: names[product_id])
    count = st.slider("Number of recommendations", 1, 6, 4)
    recommendations = engine.recommend(selected, count)
    for column, (_, product) in zip(st.columns(2) * 3, recommendations.iterrows()):
        with column:
            with st.container(border=True):
                st.caption(product["category"])
                st.subheader(product["name"])
                st.write(product["description"])
                st.write(f"?{product['price']:,.0f}")
                st.caption(f"Description similarity {product['similarity']:.0%}")
    with st.expander("Browse the full demo catalog"):
        st.dataframe(catalog, hide_index=True)
    st.caption("Prices are illustrative demo values in INR.")
    st.caption("TF-IDF and cosine similarity rank shared product language. Similarity is not purchase probability.")
with tabs[3]:
    st.subheader("What does the image model see?")
    st.write("Upload a clear photo of one object. MobileNet V3 uses pretrained ImageNet labels; unfamiliar products may be misclassified.")
    uploaded = st.file_uploader("Product image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        try:
            if uploaded.size > 10 * 1024 * 1024:
                raise ValueError("Please use an image smaller than 10 MB.")
            photo = Image.open(uploaded)
            photo.load()
            st.image(photo, width=300)
            if st.button("Analyze image", type="primary"):
                with st.spinner("Loading CNN and analyzing image..."):
                    st.dataframe(engine.classify_image(photo), hide_index=True)
                st.caption("Scores are model softmax outputs, not verified confidence. First use downloads pretrained weights.")
        except (UnidentifiedImageError, ValueError, OSError, Image.DecompressionBombError) as exc:
            st.error(f"Cannot process image: {exc}")
with tabs[4]:
    st.subheader("Listen to the customer")
    review = st.text_area("Customer review", "The headphones sound great, but delivery was slow.", max_chars=10000)
    if st.button("Analyze review", type="primary"):
        if not review.strip():
            st.warning("Enter a review first.")
        else:
            sentiment = engine.analyze_sentiment(review)
            st.subheader(f"Overall sentiment: {sentiment['label']}")
            st.metric("Sentiment score", f"{sentiment['compound']:+.2f}")
            st.caption("The score ranges from -1 (strongly negative) to +1 (strongly positive).")
            positive, neutral, negative = st.columns(3)
            positive.metric("Positive", f"{sentiment['pos']:.0%}")
            neutral.metric("Neutral", f"{sentiment['neu']:.0%}")
            negative.metric("Negative", f"{sentiment['neg']:.0%}")
            st.caption("These proportions describe the sentiment composition of the text.")
    st.caption("English VADER sentiment handles many negations and intensifiers. Sarcasm, mixed opinions and other languages remain limitations.")

