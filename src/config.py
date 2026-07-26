from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
RAW_DATA = DATA / 'raw'
PROCESSED_DATA = DATA / 'processed'
TRAIN_DATA = PROCESSED_DATA / 'train.pt'
TEST_DATA = PROCESSED_DATA / 'test.pt'
PARAMS = ROOT / 'params.yaml'