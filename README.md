# Getting Started
* Prepare Data
  * Data should be placed at mimiciii/data/final_dataset/{dataset_version}/dataset_4.csv
* Install required Python packages
```
python -m venv .venv
pip install -r requirements.txt
```
* Preprocessing Data
```
cd mimiciii/source_code/preprocess/ && python normalization.py
```
* Training
```
mkdir mimiciii/source_code/logs && cd mimiciii/source_code/logs && mlflow server --host 127.0.0.1 --port 8787
cd mimiciii/source_code/ && python train.py
```
