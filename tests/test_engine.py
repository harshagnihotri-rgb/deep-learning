from pathlib import Path
import pytest
import engine

def test_purchase_probability_and_invalid_input():
    assert 0 <= engine.predict_purchase(engine.DEFAULT_SESSION)['probability'] <= 1
    for value in [-1, float('nan'), float('inf')]:
        with pytest.raises(ValueError): engine.predict_purchase({'ProductRelated': value})

def test_recommendations_exclude_query_and_are_ranked():
    product = engine.get_catalog().iloc[0].product_id
    result = engine.recommend(product, 4)
    assert product not in result.product_id.tolist()
    assert len(result) <= 4
    assert result.similarity.is_monotonic_decreasing
    with pytest.raises(ValueError): engine.recommend(-1)

def test_negation_and_empty_review():
    assert engine.analyze_sentiment('This is good')['label'] == 'Positive'
    assert engine.analyze_sentiment('This is not good')['label'] == 'Negative'
    with pytest.raises(ValueError): engine.analyze_sentiment('   ')

def test_evaluation_counts_and_confusion_matrix():
    metrics = engine.load_metrics()
    assert metrics['train_rows'] + metrics['test_rows'] == metrics['rows']
    assert sum(sum(row) for row in metrics['confusion_matrix']) == metrics['test_rows']
    assert metrics['roc_auc'] > 0.5

def test_review_interaction():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=60)
    app.text_area[0].set_value('This is not good')
    next(b for b in app.button if b.label == 'Analyze review').click().run(timeout=60)
    assert not app.exception
    assert any('Negative' in str(j.value) for j in app.json)

def test_app_renders_and_purchase_form_works():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=60)
    assert not app.exception
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert app.metric
