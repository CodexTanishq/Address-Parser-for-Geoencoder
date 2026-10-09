"""Train v3 on non-OUT rows only, using the v2 settings unchanged."""
import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]
if (P/'models'/'ner_v3_best').exists(): raise FileExistsError('models/ner_v3_best already exists')
s=importlib.util.spec_from_file_location('v2',P/'code_files'/'04_train_ner.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.TRAIN_FILE=P/'data'/'labels'/'train_bio_merged_v2_noout.csv';m.VALIDATION_FILE=P/'data'/'labels'/'validation_bio_noout.csv';m.MODEL_DIR=P/'models'/'ner_v3_best';m.WORK_DIR=P/'temp_code'/'ner_training_v3'
if __name__=='__main__':m.main()
