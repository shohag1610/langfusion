from utils.clean_text import clean_text

def test_clean_text():
    # 1. Empty input
    assert clean_text("") == ""
    assert clean_text(None) == ""

    # 2. Leading/trailing whitespace
    assert clean_text("  hello world  ") == "hello world"

    # 3. Multiple spaces/newlines
    input_text = "Hello   \n   world \n\n test"
    expected = "Hello world test"
    assert clean_text(input_text) == expected

    # 4. Escape quotes
    input_text = 'She said "Hello" to him'
    expected = 'She said \\"Hello\\" to him'
    assert clean_text(input_text) == expected

    # 5. Remove control characters
    input_text = "Hello\x00\x1FWorld\x7F!"
    expected = "Hello World!"
    assert clean_text(input_text) == expected

    # 6. Combination
    input_text = '  \nTest "string"\x00 with   spaces \n\n '
    expected = 'Test \\"string\\" with spaces'
    assert clean_text(input_text) == expected