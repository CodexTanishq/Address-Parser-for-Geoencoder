# Fuzzy Spelling Evaluation Report (N=10)

## Summary
This evaluation demonstrates the effect of length-based Levenshtein matching, abbreviation mapping, and town-scoped multi-candidate resolution on 10 hand-crafted noisy Indian address queries.

## Comparative Results (Before vs After)

| ID | Query String | Expected Target (Town, Loc, LM) | Before (Baseline Labeller) | After (Enhanced v2 Model) | POI Resolution Mode & Candidates |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F01** | `silver oak lyt 4th cross 8th main nr govt school 980301` | **Navanagara East** / **Silver Oak Layout** / **govt_school** | Town: Navanagara East, Loc: Silver Oak Layout, LM: govt_school | **Town: Navanagara East, Loc: Silver Oak Layout, LM: govt_school** | `exact_key` (1 candidate) |
| **F02** | `devggarh ngr gali 4 near church patel nagar 970202` | **Devgarh Nagar** / **Patel Nagar** / **church** | Town: Devgarh Nagar, Loc: Patel Nagar, LM: church | **Town: Devgarh Nagar, Loc: Patel Nagar, LM: church** | `exact_key` (1 candidate) |
| **F03** | `10th cross gandhi nagr hanuman mandr kaveripura 960102` | **Kaveripura** / **Gandhi Nagar** / **hanuman_temple** | Town: Kaveripura, Loc: Gandhi Nagar, LM: NONE | **Town: Kaveripura, Loc: Gandhi Nagar, LM: hanuman_temple** | `exact_key` (1 candidate) |
| **F04** | `kuvempu layt 3rd cross opp bus stand kaveripura 960102` | **Kaveripura** / **Kuvempu Layout** / **bus_stop** | Town: Kaveripura, Loc: Kuvempu Layout, LM: bus_stop | **Town: Kaveripura, Loc: Kuvempu Layout, LM: bus_stop** | `exact_key` (1 candidate) |
| **F05** | `house 12 indira nagr nr ration shop devgarh ngr 970201` | **Devgarh Nagar** / **Indira Colony** / **ration_shop** | Town: Devgarh Nagar, Loc: Indira Colony, LM: ration_shop | **Town: Devgarh Nagar, Loc: Indira Colony, LM: ration_shop** | `candidates` (9 candidates) |
| **F06** | `7th mn rd 2nd x ashoka layt kaveripura 960104` | **Kaveripura** / **Ashoka Layout** / **NONE** | Town: Kaveripura, Loc: Ashoka Layout, LM: NONE | **Town: Kaveripura, Loc: Ashoka Layout, LM: NONE** | N/A (No landmark) |
| **F07** | `bhd medicals basava nagr kaveripura 960101` | **Kaveripura** / **Basava Nagar** / **medical_store** | Town: Kaveripura, Loc: Basava Nagar, LM: medical_store | **Town: Kaveripura, Loc: Basava Nagar, LM: medical_store** | `exact_key` (1 candidate) |
| **F08** | `5th cross vinayaka layt nr pani ki tanki 960104` | **Kaveripura** / **Vinayaka Nagar** / **water_tank** | Town: Kaveripura, Loc: Vinayaka Nagar, LM: water_tank | **Town: Kaveripura, Loc: Vinayaka Nagar, LM: water_tank** | `candidates` (4 candidates) |
| **F09** | `opp doodh dairy nehru colny devgarh ngr 970203` | **Devgarh Nagar** / **Nehru Colony** / **milk_dairy** | Town: Devgarh Nagar, Loc: Nehru Colony, LM: milk_dairy | **Town: Devgarh Nagar, Loc: Nehru Colony, LM: milk_dairy** | `candidates` (5 candidates) |
| **F10** | `nr kalyana mantapa shanthi nagr navanagara 980302` | **Navanagara East** / **UNKNOWN** / **community_hall** | Town: Navanagara East, Loc: UNKNOWN, LM: community_hall | **Town: Navanagara East, Loc: UNKNOWN, LM: community_hall** | `candidates` (4 candidates) |

## Key Observations & Enhancements Validated
1. **Colloquial Abbreviation Expansion**: Abbreviation tokens such as `lyt`/`layt` $\to$ `layout`, `ngr` $\to$ `nagar`, `colny`/`col` $\to$ `colony`, `mn rd` $\to$ `main road`, and `x` $\to$ `cross` are transparently canonicalized.
2. **Typo / Slang Landmark Recovery**: In query **F03**, `hanuman mandr` was previously missed completely by baseline string matching. With length-conditioned Levenshtein distance ($len=6, dist=1$), it is resolved to `hanuman_temple`.
3. **Multi-Candidate POI Dispersion**: In queries **F05**, **F08**, **F09**, and **F10**, where the specific locality either lacks an exact key for the landmark or is ambiguous, the system emits all candidate POIs of that type in the resolved town (`candidates`) rather than guessing a single unverified coordinate.
