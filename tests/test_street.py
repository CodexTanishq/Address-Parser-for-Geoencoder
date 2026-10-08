import pytest
from src.street import extract_streets


def test_multiple_streets_not_combined():
    # 6th Cross, 5th Main must generate two distinct STREET spans
    text = "6th Cross, 5th Main, Church Hattira, Kuvempu Layt"
    spans = extract_streets(text)
    assert len(spans) == 2
    assert spans[0].text.lower() == "6th cross"
    assert spans[1].text.lower() == "5th main"


def test_street_abbreviations():
    # 7th Cr, 7th Main
    spans = extract_streets("231/2, 7th Cr, 7th Main, Nehru Colony")
    assert len(spans) == 2
    assert "7th Cr" in [s.text for s in spans]

    # 9th crs, 8th mn
    spans = extract_streets("HOUSE 179, 9TH CRS, 8TH MN, KAVERIPURA")
    assert len(spans) == 2

    # 6th X, 1st Main
    spans = extract_streets("6th X, 1st Main, nr Medical Store")
    assert len(spans) == 2


def test_road_and_gali_formats():
    # Rd 6 and Road 5
    spans = extract_streets("HOUSE 32, BLK C, RD 6, NAVANAGARA EAST")
    assert len(spans) == 1
    assert spans[0].text.upper() == "RD 6"

    # 2nd road
    spans = extract_streets("367/3 e block 2nd road green park layt")
    assert any(s.text.lower() == "2nd road" for s in spans)

    # Gali 10
    spans = extract_streets("#81 gali 10 ganesh mandir")
    assert len(spans) == 1
    assert spans[0].text.lower() == "gali 10"

    # Gali no-4
    spans = extract_streets("217/2, Gali no-4, opp Water Tank")
    assert len(spans) == 1
    assert spans[0].text.lower() == "gali no-4"

    # Devanagari gali
    spans = extract_streets("गली नं. 3, राशन की दुकान के बगल में")
    assert len(spans) == 1
    assert "गली नं. 3" in spans[0].text
