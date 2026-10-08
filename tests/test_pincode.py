import pytest
from src.pincode import extract_pincode
from src import config


def test_valid_pincodes():
    # Standard 6-digit pincode
    pin, span = extract_pincode("Kaveripura - 960102")
    assert pin == "960102"
    assert span is not None
    assert span.entity_type == "PINCODE"
    assert span.confidence == config.CONF_HIGH


def test_non_pincodes_rejected():
    # Ordinals like 6th must NOT be extracted as pincode
    pin, span = extract_pincode("6th Cross, 5th Main")
    assert pin is None
    assert span is None

    # Slash numbers like 217/2 must NOT be extracted as pincode
    pin, span = extract_pincode("217/2, Gali no-4")
    assert pin is None

    # 5-digit number must NOT be extracted
    pin, span = extract_pincode("Plot 96010")
    assert pin is None

    # 7-digit number must NOT be extracted
    pin, span = extract_pincode("Phone 1234567")
    assert pin is None

    # 7-digit invalid pincode test case in data
    pin, span = extract_pincode("#159, 6th Cross, 7th Main, Kaveripura - 9601003")
    assert pin is None


def test_ambiguous_multiple_pincodes():
    # Two conflicting valid 6-digit pincodes in the same address
    pin, span = extract_pincode("Moved from 960102 to 960104")
    assert span is not None
    assert span.confidence == config.CONF_AMBIGUOUS


def test_pincode_town_cross_reference():
    valid_town_pins = {"960101", "960102", "960103", "960104"}
    # In-town pincode
    pin, span = extract_pincode("Kaveripura 960102", town_id="T1", valid_pincodes_for_town=valid_town_pins)
    assert span.confidence == config.CONF_HIGH

    # Disagreeing pincode
    pin, span = extract_pincode("Kaveripura 970202", town_id="T1", valid_pincodes_for_town=valid_town_pins)
    assert span.confidence == config.CONF_MEDIUM
