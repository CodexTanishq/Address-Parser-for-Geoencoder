import sys
import io

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.utils import load_raw_data
from src.parser import AddressParser
from src.inference import AddressNERPredictor

# Selected diverse representative test cases directly hardcoded from data/addresses.csv
# Format: (address_id, town_id, raw_address, test_category)
TEST_ADDRESSES = [
    # 1. Kannada Script & Multilingual Cues
    ("AD000001", "T1", "6th Cross, 5th Main, ಚರ್ಚ್ ಹತ್ತಿರ, Kuvempu Layt, Kaveripura - 960102", "Kannada Script + Suffix Landmark"),
    ("AD000048", "T1", "#160, 5th Cross, 1st Main, ಕಲ್ಯಾಣ ಮಂಟಪ ಹತ್ತಿರ, Kalyana Ngr, Kaveripura - 960104", "Kannada Script + Kalyana Mantapa POI"),
    ("AD000085", "T1", "house 283 ಬಸ್ ನಿಲ್ದಾಣ ಹತ್ತಿರ gandhi nagar kaveripura - 960102", "Kannada Bus Stand POI + Lowercase"),
    ("AD000102", "T1", "#266 6th crs 7th main ನ್ಯಾಯಬೆಲೆ ಅಂಗಡಿ ಪಕ್ಕ anjaneya badavane kaveripura - 960103", "Kannada PDS Shop + Pakka (Beside)"),

    # 2. Devanagari Script & Multilingual Cues
    ("AD000007", "T2", "गली नं. 3, राशन की दुकान के बगल में, Tilak Ngr, Devgarh Nagar - 970204", "Devanagari Street & Landmark + Tilak Ngr"),
    ("AD000012", "T2", "house 52 gali no. 5 ward 9 मस्जिद के पास shastri nagar devgarh nagar - 970201", "Devanagari Mosque Landmark + Ward 9"),
    ("AD000069", "T2", "No. 124, Gali no-3, डाकघर के पास, Shivaji Nagar, Devgarh Ngr - 970202", "Devanagari Post Office POI"),
    ("AD000089", "T2", "77/3, गली नं. 9, Ward 3, हनुमान मंदिर के पास, Ambedkar Nagar, Devgarh Ngr - 970204", "Devanagari Hanuman Temple + Slash HNo"),

    # 3. Varied House Number Formats
    ("AD000004", "T2", "#81 gali 10 ganesh mandir ke bagal mein krishna puri devgarh nagar - 970202", "#81 House Number + ke bagal mein"),
    ("AD000017", "T2", "217/2, Gali no-4, opp. Water Tank, Ambedkar Nagr, Devgarh Nagar - 970204", "217/2 Slash HNo + Gali no-4 + opp POI"),
    ("AD000040", "T1", "313/2, 3rd Crs, 8th Main, Basava Nagar, Kaveripura", "313/2 Slash HNo + Street Abbrevs"),
    ("AD000027", "T2", "89/2, Azad Mohalla, Devgarh Nagar - 970203", "89/2 Slash HNo + Standalone Locality"),
    ("AD000002", "T1", "no. 173 12th cross 3rd main shanthi nagar kaveripura - 960101", "no. 173 Prefixed HNo + Lowercase"),
    ("AD000015", "T1", "H.NO. 356, NR. CHURCH, KUVEMPU LAYOUT, KAVERIPURA - 960104", "H.NO. 356 Uppercase + nr Church"),
    ("AD000021", "T3", "HOUSE 32, BLK C, RD 6, NAVANAGARA EAST - 980303", "HOUSE 32 + BLK C + RD 6"),

    # 4. Street Patterns & Combinations
    ("AD000292", "T1", "no. 404 6th cross 5th main nr. ganapathi temple anjaneya badavane kaveripura - 960103", "6th Cross + 5th Main (Separate Streets)"),
    ("AD000318", "T1", "6th cross 5th main close to pharmacy kuvempu layout kaveripura - 960102", "Multi-Street + close to landmark"),
    ("AD000010", "T1", "6th X, 1st Main, nr Medical Store, Kaveripura - 960104", "6th X (Cross Abbrev) + Medical Store"),
    ("AD000014", "T2", "HOUSE 458, GALI NO-4, AMBEDKAR NAGAR, DEVGARH NAGAR", "Gali no-4 Uppercase without Pincode"),
    ("AD000013", "T2", "H.No. 186, Gali No. 4, Ward 3, opp Overhead Tank, Ambedkar Nagar, Devgarh Nagar - 970204", "Gali No. 4 + Ward 3 Separation"),
    ("AD000016", "T2", "No. 86, Gali 2, Ward 8, Azad Mohalla, Devgarh Nagar - 970203", "Ward 8 + Gali 2 + No. 86"),
    ("AD000011", "T3", "blk a rd 2 milk dairy edurru palm meadows navanagara east - 980302", "blk a + rd 2 + edurru landmark"),
    ("AD000019", "T3", "#421 blk d rd 6 near water tank silver oak layt navanagara east - 980301", "blk d rd 6 + near water tank"),

    # 5. Multilingual Landmark Indicators & POIs
    ("AD000327", "T1", "2nd Cross, 7th Mn, Ration ki Dukaan ke bagal mein, Kuvempu Layt, Kaveripura - 960103", "Ration ki Dukaan ke bagal mein"),
    ("AD001385", "T3", "#461, F BLOCK, 4TH ROAD, COMMUNITY HALL KE BAGAL MEIN, UNITY NAGAR, NAVANAGARA EAST - 980302", "Community Hall ke bagal mein Uppercase"),
    ("AD000039", "T2", "Sai Residency, Gali 2, Dawai ki Dukaan ke paas, Ambedkar Nagar, Devgarh Nagar - 970204", "Dawai ki Dukaan ke paas"),
    ("AD000052", "T1", "10th cross 1st main hanuman mandir ke paas gandhi nagr kaveripura - 960102", "Hanuman mandir ke paas + gandhi nagr"),
    ("AD000034", "T1", "#475 ganapathi gudi hattira kuvempu layout kaveripura - 960102", "ganapathi gudi hattira"),
    ("AD000047", "T1", "House 353, 5th X, 8th Main, Sarkari Shaale hattira, Vinayaka Nagar, Kaveripura - 960104", "Sarkari Shaale hattira POI"),
    ("AD000086", "T2", "h.no. 107 gali no. 5 opp water tank krishna puri devgarh nagar", "opp water tank + No Pincode"),
    ("AD000005", "T2", "behind PDS Shop, Shivaji Nagar, Devgarh Nagar - 970202", "behind PDS Shop leading landmark"),
    ("AD000035", "T3", "A Block, 2nd Road, behind Milk Dairy, Palm Meadows, Navanagara East - 980302", "behind Milk Dairy POI"),

    # 6. Misspelled & Abbreviated Localities
    ("AD000050", "T1", "197/1, 4TH CROSS, 8TH MAIN, KUVEMPU LAYT, KAVERIPURA - 960104", "Kuvempu Layt Abbreviation"),
    ("AD000023", "T3", "h.no. 64 blk d rd 6 silver oak layout navanagara east - 980301", "silver oak layout lowercase"),
    ("AD000057", "T3", "441/3, NR. GOVT SCHOOL, SILVUR OAK LAYOUT, NAVANAGARA EAST - 980301", "SILVUR OAK Spelling Variant"),
    ("AD000028", "T3", "Blk F, Rd 4, Defence Col, Navanagara East - 980302", "Defence Col Abbreviation"),
    ("AD000042", "T3", "House 217, Lakevew Phase 2, Navanagara East - 980303", "Lakevew Phase 2 Typo"),
    ("AD000076", "T2", "Gali 9, Ward 7, close to Ganesh Temple, Gandhi Baste, Devgarh Nagr - 970203", "Gandhi Baste Typo + Devgarh Nagr"),

    # 7. Missing Components & Complex Cases
    ("AD000020", "T1", "house 25 3rd cross 6th main rd kaveripura - 960104", "Missing Locality (Direct to Town)"),
    ("AD000018", "T1", "#159, 6th Cross, 7th Main, Anjaneya Badavane, Kaveripura - 9601003", "Invalid 7-digit Pincode 9601003"),
    ("AD000062", "T1", "H.No. 339, 6th Cr, 3rd Main, Bus Nildana eduru, Gandhi Nagar, Kaveripura", "Missing Pincode + Bus Nildana eduru"),

    # 8. Clean Standard Addresses
    ("AD000026", "T1", "H.No. 377, 9th Cross, 7th Main, Vinayaka Nagar, Kaveripura - 960104", "Standard Well-Formed Address"),
]


def format_field(val):
    if val is None or val == "":
        return "None"
    if isinstance(val, list):
        return ", ".join(val) if val else "None"
    return str(val)


def run_stress_test():
    print("=" * 70)
    print("  QUALITATIVE STRESS TEST: ADDRESS PARSER + MuRIL NER MODEL")
    print("=" * 70)
    print(f"Total test cases selected: {len(TEST_ADDRESSES)}\n")

    # Load resources for deterministic parser and production inference engine
    try:
        _, towns_df, localities_df = load_raw_data()
        parser = AddressParser(towns_df, localities_df)
    except Exception as e:
        print(f"CRITICAL: Failed to initialize AddressParser: {e}")
        return

    try:
        predictor = AddressNERPredictor()
    except Exception as e:
        print(f"CRITICAL: Failed to initialize AddressNERPredictor: {e}")
        return

    success_count = 0

    for idx, (addr_id, town_id, raw_address, category) in enumerate(TEST_ADDRESSES, start=1):
        print("=" * 70)
        print(f"TEST {idx:02d} | ID: {addr_id} | Category: {category}")
        print("=" * 70)
        print(f"RAW:\n{raw_address}\n")

        # 1. Deterministic Parser Output
        try:
            parsed = parser.parse(address_id=addr_id, address_text=raw_address, town_id=town_id)
            print("PARSER OUTPUT:")
            print(f"  HOUSE_NUMBER : {format_field(parsed.house_number)}")
            print(f"  STREET       : {format_field(parsed.streets)}")
            print(f"  WARD         : {format_field(parsed.wards)}")
            print(f"  LANDMARK     : {format_field(parsed.landmarks)}")
            print(f"  LOCALITY     : {format_field(parsed.locality_name)}")
            print(f"  TOWN         : {format_field(parsed.town_name)}")
            print(f"  PINCODE      : {format_field(parsed.pincode)}")
            print(f"  OTHER        : {format_field(parsed.other_details)}")
        except Exception as e:
            print(f"PARSER ERROR: {e}")

        # 2. MuRIL NER Output
        try:
            ner_res = predictor.predict(raw_address)
            struct = ner_res["structured_address"]
            print("\nMURIL OUTPUT:")
            print(f"  HOUSE_NUMBER : {format_field(struct['house_number'])}")
            print(f"  STREET       : {format_field(struct['streets'])}")
            print(f"  WARD         : {format_field(struct['ward'])}")
            print(f"  LANDMARK     : {format_field(struct['landmarks'])}")
            print(f"  LOCALITY     : {format_field(struct['locality'])}")
            print(f"  TOWN         : {format_field(struct['town'])}")
            print(f"  PINCODE      : {format_field(struct['pincode'])}")

            # Print token-level BIO tags
            tokens = ner_res.get("tokens", [])
            bio_tags = ner_res.get("bio_tags", [])
            if tokens and bio_tags:
                non_o_tokens = [(t, tag) for t, tag in zip(tokens, bio_tags) if tag != "O"]
                if non_o_tokens:
                    token_str = " | ".join([f"{t} [{tag}]" for t, tag in non_o_tokens])
                    print(f"\n  PREDICTED ENTITY TOKENS:\n    {token_str}")
        except Exception as e:
            print(f"MURIL ERROR: {e}")

        print("=" * 70 + "\n")
        success_count += 1

    print("TESTING COMPLETE")
    print(f"Total addresses tested: {success_count}")


if __name__ == "__main__":
    run_stress_test()
