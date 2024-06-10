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
cd mimiciii/source_code/ && python train.py
```
