import pytest
from src.house_number import extract_house_number
from src.street import extract_streets
from src.ward import extract_wards


def test_explicit_house_numbers():
    # H.No. 356
    val, span = extract_house_number("H.No. 356, NR. CHURCH, KUVEMPU LAYOUT")
    assert val == "356"
    assert span is not None
    assert span.text == "356"

    # #81
    val, span = extract_house_number("#81 Gali 10 Ganesh Mandir ke bagal mein")
    assert val == "81"
    assert span.text == "81"

    # 217/2
    val, span = extract_house_number("217/2, Gali no-4, opp Water Tank")
    assert val == "217/2"
    assert span.text == "217/2"

    # HOUSE 32
    val, span = extract_house_number("HOUSE 32, BLK C, RD 6")
    assert val == "32"
    assert span.text == "32"

    # NO. 173
    val, span = extract_house_number("no. 173 12th cross 3rd main shanthi nagar kaveripura")
    assert val == "173"
    assert span.text == "173"


def test_do_not_extract_ordinals():
    # 6th Cross, 5th Main must NOT yield 6th or 5th as house number
    text = "6th Cross, 5th Main, Church Hattira, Kuvempu Layt"
    streets = extract_streets(text)
    val, span = extract_house_number(text, protected_spans=streets)
    assert val is None
    assert span is None


def test_do_not_extract_street_or_ward_numbers():
    # 'Gali 10' or 'Gali no. 4' must not have '10' or '4' as house number
    text = "Gali no. 4, Ward 3, Ambedkar Nagar"
    streets = extract_streets(text)
    wards = extract_wards(text)
    val, span = extract_house_number(text, protected_spans=streets + wards)
    assert val is None
    assert span is None
