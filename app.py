"""Retail intelligence dashboard. Run: streamlit run app.py."""
import streamlit as st
from PIL import Image, UnidentifiedImageError
import engine

st.set_page_config(page_title="RetailLens | Commerce Intelligence", page_icon="ðŸ›ï¸", layout="wide")
st.markdown("""<style>.stApp{background:#f5f7fb} h1,h2,h3{color:#102b46} [data-testid=stMetric]{background:white;border:1px solid #dce5ee;padding:18px;border-radius:14px} .block-container{padding-top:2rem} </style>""", unsafe_allow_html=True)
st.caption("RETAILLENS / E-COMMERCE INTELLIGENCE")
st.title("Turn shopping signals into insights.")
st.write("Explore customer intent, discover related products, understand images, and listen to customer feedback.")
with st.sidebar:
    st.header("RetailLens")
    st.write("An explainable academic AI project")
    st.info("Predictions support exploration. They are not guarantees of customer behaviour.")
    st.caption("Purchase: real UCI sessions Â· Recommendations: demonstration catalog Â· Images: pretrained ImageNet CNN Â· Sentiment: VADER rules")
tabs = st.tabs(["Overview", "Purchase intent", "Similar products", "Image lab", "Review sentiment"])
with tabs[0]:
    st.subheader("Four tools. One workspace.")
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
            st.json(metrics)
    except (FileNotFoundError, RuntimeError) as exc:
        st.warning(f"Train the purchase model first: python scripts/train.py. {exc}")
    st.warning("Image labels use ImageNet's categories, not the four custom categories in the original proposal. Product recommendations are content based, not personalized to an individual.")
with tabs[1]:
    st.subheader("Will this browsing session convert?")
    st.write("Uses session browsing summaries; live early-session predictions are not validated. PageValues is excluded to reduce leakage risk.")
    with st.form("purchase"):
        left, right = st.columns(2)
        with left:
            admin = st.number_input("Administrative pages", 0, 100, 1)
            admin_time = st.number_input("Administrative duration (seconds)", 0.0, 100000.0, 30.0)
            info = st.number_input("Informational pages", 0, 100, 0)
            info_time = st.number_input("Informational duration (seconds)", 0.0, 100000.0, 0.0)
            product = st.number_input("Product pages", 0, 1000, 15)
            product_time = st.number_input("Product duration (seconds)", 0.0, 100000.0, 500.0)
            bounce = st.slider("Bounce rate", 0.0, 1.0, 0.02)
            exits = st.slider("Exit rate", 0.0, 1.0, 0.04)
        with right:
            special = st.slider("Closeness to a special day", 0.0, 1.0, 0.0)
            month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
            visitor = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
            weekend = st.checkbox("Weekend")
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
            st.caption("Model probability is an estimate; validation on your store is required before business use.")
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
with tabs[2]:
    st.subheader("Find the next relevant product")
    catalog = engine.get_catalog()
    st.dataframe(catalog, hide_index=True, use_container_width=True)
    selected = st.selectbox("Select product ID", catalog['product_id'].tolist())
    count = st.slider("Number of recommendations", 1, 6, 4)
    st.dataframe(engine.recommend(selected, count), hide_index=True, use_container_width=True)
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
            st.json(engine.analyze_sentiment(review))
    st.caption("English VADER sentiment handles many negations and intensifiers. Sarcasm, mixed opinions and other languages remain limitations.")

