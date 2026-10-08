import pytest
import pandas as pd
from src.parser import AddressParser
from src.bio import spans_to_bio, validate_bio_sequence
from src import config


@pytest.fixture
def parser():
    towns_df = pd.read_csv(config.TOWNS_CSV)
    localities_df = pd.read_csv(config.LOCALITIES_CSV)
    return AddressParser(towns_df, localities_df)


def test_canonical_bio_alignment(parser):
    text = "6th Cross, 5th Main, Church Hattira, Kuvempu Layt, Kaveripura - 960102"
    parsed = parser.parse("AD_BIO", text, "T1")

    tokens, labels, meta = spans_to_bio(text, parsed.spans, confidence_filter=config.CONF_HIGH)

    token_label_map = dict(zip(tokens, labels))

    # Verify key tokens
    assert token_label_map["6th"] == "B-STREET"
    assert token_label_map["Cross"] == "I-STREET"
    assert token_label_map["5th"] == "B-STREET"
    assert token_label_map["Main"] == "I-STREET"
    assert token_label_map["Church"] == "B-LANDMARK"
    assert token_label_map["Hattira"] == "I-LANDMARK"
    assert token_label_map["Kuvempu"] == "B-LOCALITY"
    assert token_label_map["Layt"] == "I-LOCALITY"
    assert token_label_map["Kaveripura"] == "B-TOWN"
    assert token_label_map["960102"] == "B-PINCODE"
    assert token_label_map[","] == "O"
    assert token_label_map["-"] == "O"

    # Verify validity of transitions
    assert validate_bio_sequence(labels) is True


def test_no_overlapping_bio_labels(parser):
    text = "#81 Gali 10 Ganesh Mandir ke bagal mein krishna puri devgarh nagar - 970202"
    parsed = parser.parse("AD_BIO2", text, "T2")

    tokens, labels, meta = spans_to_bio(text, parsed.spans, confidence_filter=config.CONF_HIGH)

    assert validate_bio_sequence(labels) is True
    # Verify entity assignments
    token_label_map = dict(zip(tokens, labels))
    assert token_label_map["81"] == "B-HOUSE_NUMBER"
    assert token_label_map["Gali"] == "B-STREET"
    assert token_label_map["10"] == "I-STREET"
    assert token_label_map["970202"] == "B-PINCODE"
