import pandas as pd
from src.data_codec import encoder  # adjust import path


def test_encoder_with_news_data(tmp_path):
    # Arrange: your provided dataset
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

    output_file = tmp_path / "encoded.csv"

    # Act
    encoded_df, encoders = encoder(df, encoded_path=output_file)

    # Assert correct datatypes
    assert encoded_df["label"].dtype == "int64"
    assert encoded_df["title"].dtype == "int64"
    assert encoded_df["description"].dtype == "int64"

    # Assert encoders exist
    assert set(encoders.keys()) == {"label", "title", "description"}

    # Assert file created
    assert output_file.exists()
