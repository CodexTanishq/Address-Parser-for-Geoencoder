"""Run the v1 NER workflow with v2 labels and v2 output locations."""
import importlib.util
from pathlib import Path

P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v1_train',P/'code_files'/'04_train_ner.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.TRAIN_FILE=P/'data'/'labels'/'train_bio_merged_v2.csv'
m.MODEL_DIR=P/'models'/'ner_v2'
m.WORK_DIR=P/'temp_code'/'ner_training_v2'
if __name__=='__main__': m.main()
