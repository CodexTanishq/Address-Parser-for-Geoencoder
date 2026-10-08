import pytest
import pandas as pd
from src.locality import LocalityResolver
from src import config


@pytest.fixture
def locality_resolver():
    localities_df = pd.read_csv(config.LOCALITIES_CSV)
    return LocalityResolver(localities_df)


def test_abbreviation_and_spelling_matching(locality_resolver):
    # Kuvempu Layt -> Kuvempu Layout
    loc, span, meta = locality_resolver.resolve("6th Cross, Kuvempu Layt, Kaveripura", town_id="T1")
    assert loc == "Kuvempu Layout"
    assert span is not None
    assert span.entity_type == "LOCALITY"

    # silver oak layou -> Silver Oak Layout
    loc, span, meta = locality_resolver.resolve("#421 near water tank silver oak layou navanagara east", town_id="T3")
    assert loc == "Silver Oak Layout"

    # gandhi nagr -> Gandhi Nagar in T1
    loc, span, meta = locality_resolver.resolve("house 283 gandhi nagr kaveripura", town_id="T1")
    assert loc == "Gandhi Nagar"

    # Defence Col -> Defence Colony
    loc, span, meta = locality_resolver.resolve("Blk F, Rd 4, Defence Col, Navanagara East", town_id="T3")
    assert loc == "Defence Colony"

    # Lakevew Phase 2 -> Lakeview Phase 2
    loc, span, meta = locality_resolver.resolve("House 217, Lakevew Phase 2, Navanagara East", town_id="T3")
    assert loc == "Lakeview Phase 2"


def test_town_scoping(locality_resolver):
    # Gandhi Basti in T2
    loc_t2, span_t2, meta_t2 = locality_resolver.resolve("H.No. 206, Gandhi Basti, Devgarh Nagar", town_id="T2")
    assert loc_t2 == "Gandhi Basti"
    assert meta_t2["locality_id"] == "T2-L07"

    # Gandhi Nagar in T1
    loc_t1, span_t1, meta_t1 = locality_resolver.resolve("house 283 gandhi nagr kaveripura", town_id="T1")
    assert loc_t1 == "Gandhi Nagar"
    assert meta_t1["locality_id"] == "T1-L04"

    # Nehru Colony exists in BOTH T1 and T2: town_id correctly scopes the locality_id
    _, _, meta_nehru_t1 = locality_resolver.resolve("Nehru Colony, Kaveripura", town_id="T1")
    assert meta_nehru_t1["locality_id"] == "T1-L11"

    _, _, meta_nehru_t2 = locality_resolver.resolve("Nehru Colony, Devgarh Nagar", town_id="T2")
    assert meta_nehru_t2["locality_id"] == "T2-L06"


def test_no_false_positive_when_locality_missing(locality_resolver):
    # Addresses with NO locality mentioned must return None (never guess!)
    text = "HOUSE 32, BLK C, RD 6, NAVANAGARA EAST - 980303"
    loc, span, meta = locality_resolver.resolve(text, town_id="T3")
    assert loc is None
    assert span is None
