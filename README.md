# NLP_pipeline
```
project/
│
├── nlp_prep.py                    # main preprocessing pipeline
├── semantic.py                    # sentiment pipeline
│
├── nlp_config.py                  # all constants & dictionaries
├── nlp_features.py                # tagging, topics, detection logic
│
└── data/
    ├── input.csv                  # outputs of the scrappers
    ├── preprocessed.parquet       # outputs of nlp_prep.py
    └── semantic.parquet           # outputs of the semantic.py
```
