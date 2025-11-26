import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess_dataset import preprocess  # adjust import path


def test_preprocess_creates_text_column():
    df = pd.DataFrame({
        "label": [3],
        "title": ["Oil rises"],
        "description": ["Prices increase today"]
    })

    X, y, vectorizer = preprocess(df)

    # The combined text should equal title + space + description
    assert df["text"].iloc[0] == "Oil rises Prices increase today"

    # Should vectorise without error
    assert X.shape[0] == 1
    assert list(y) == [3]


def test_preprocess_with_real_news_dataset():
    # Arrange: your supplied dataset
    df = pd.DataFrame({
        "label": [3, 3, 3, 3],
        "title": [
            "Wall St. Bears Claw Back Into the Black (Reuters)",
            "Carlyle Looks Toward Commercial Aerospace (Reuters)",
            "Oil and Economy Cloud Stocks' Outlook (Reuters)",
            "Iraq Halts Oil Exports from Main Southern Pipeline (Reuters)"
        ],
        "description": [
            "Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.",
            "Reuters - Private investment firm Carlyle Group, which has a reputation for making well-timed and occasionally controversial plays in the defense industry, has quietly placed its bets on another part of the market.",
            "Reuters - Soaring crude prices plus worries about the economy and the outlook for earnings are expected to hang over the stock market next week during the depth of the summer doldrums.",
            "Reuters - Authorities have halted oil export flows from the main pipeline in southern Iraq after intelligence showed a rebel militia could strike infrastructure, an oil official said on Saturday."
        ]
    })

    # Act
    X, y, vectorizer = preprocess(df)

    # Assert: 4 samples
    assert X.shape[0] == 4

    # Labels correct
    assert list(y) == [3, 3, 3, 3]

    # Check vectorizer type
    assert isinstance(vectorizer, TfidfVectorizer)

    # Vocabulary should include words from BOTH title + description fields
    vocab = vectorizer.vocabulary_

    expected_words = [
        "reuters", "oil", "economy", "stocks", "pipeline", "iraq", "prices", "market"
    ]

    for word in expected_words:
        assert word in vocab, f"Expected '{word}' in vocabulary"