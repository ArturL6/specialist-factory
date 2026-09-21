from specialist_factory.models.components import text_features


def test_text_features_are_deterministic():
    assert text_features("Where is my package?", 32).tolist() == text_features("Where is my package?", 32).tolist()
