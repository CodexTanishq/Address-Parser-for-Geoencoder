import pytest
import pandas as pd
from src.parser import AddressParser
from src.landmark import extract_landmarks
from src.utils import AddressSpan
from src import config


@pytest.fixture
def parser():
    towns_df = pd.read_csv(config.TOWNS_CSV)
    localities_df = pd.read_csv(config.LOCALITIES_CSV)
    return AddressParser(towns_df, localities_df)


def test_landmark_contextual_patterns(parser):
    # nr church
    parsed = parser.parse("AD1", "H.No. 356, NR. CHURCH, KUVEMPU LAYOUT, KAVERIPURA - 960104", "T1")
    assert any("church" in l.lower() for l in parsed.landmarks)

    # Ganesh Mandir ke bagal mein
    parsed = parser.parse("AD2", "#81 Gali 10 Ganesh Mandir ke bagal mein krishna puri devgarh nagar - 970202", "T2")
    assert any("ganesh mandir ke bagal mein" in l.lower() for l in parsed.landmarks)

    # Church Hattira
    parsed = parser.parse("AD3", "6th Cross, 5th Main, Church Hattira, Kuvempu Layt, Kaveripura - 960102", "T1")
    assert any("church hattira" in l.lower() for l in parsed.landmarks)

    # opp Water Tank
    parsed = parser.parse("AD4", "217/2, Gali no-4, opp Water Tank, Ambedkar Nagar, Devgarh Nagar - 970204", "T2")
    assert any("water tank" in l.lower() for l in parsed.landmarks)

    # Ganapathi Gudi hattira
    parsed = parser.parse("AD5", "Ganapathi Gudi hattira, Vinayaka Nagar, Kaveripura", "T1")
    assert any("ganapathi gudi hattira" in l.lower() for l in parsed.landmarks)

    # Tanki ke pass
    parsed = parser.parse("AD6", "Gali 2, Ward 3, Tanki ke pass, Ambedkar Nagar, Devgarh Ngr - 970204", "T2")
    assert any("tanki ke pass" in l.lower() for l in parsed.landmarks)

    # Masjid ke paas
    parsed = parser.parse("AD7", "house 52 gali no. 5 ward 9 Masjid ke paas shastri nagar devgarh nagar - 970201", "T2")
    assert any("masjid ke paas" in l.lower() for l in parsed.landmarks)

    # Dawai ki Dukaan ke paas
    parsed = parser.parse("AD8", "Gali 1, Dawai ki Dukaan ke paas, Shastri Nagar, Devgarh Nagar", "T2")
    assert any("dawai ki dukaan ke paas" in l.lower() for l in parsed.landmarks)


def test_landmark_does_not_absorb_other_entities(parser):
    # CRITICAL: Landmark must NOT absorb 6th Cross, 5th Main, Kuvempu Layt, Kaveripura, 960102
    text = "6th Cross, 5th Main, Church Hattira, Kuvempu Layt, Kaveripura - 960102"
    parsed = parser.parse("AD_TEST", text, "T1")

    # Streets
    assert len(parsed.streets) == 2
    assert "6th Cross" in parsed.streets
    assert "5th Main" in parsed.streets

    # Landmark
    assert len(parsed.landmarks) == 1
    assert parsed.landmarks[0] == "Church Hattira"
    # Verify landmark text does NOT contain 6th Cross or Kuvempu
    assert "6th Cross" not in parsed.landmarks[0]
    assert "Kuvempu" not in parsed.landmarks[0]

    # Locality & Town & Pincode
    assert parsed.locality_name == "Kuvempu Layout"
    assert parsed.town_name == "Kaveripura"
    assert parsed.pincode == "960102"
