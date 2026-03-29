# NLP_pipeline
```
project/
│
├── nlp_config.py                  # all constants & dictionaries
├── nlp_features.py                # tagging, topics, detection logic
│
├── nlp_prep.py                    # main preprocessing part of the pipeline
├── nlp_semantic.py                # sentiment part of the pipeline
│
├── nlp_dashboard.py               # dashboard preview
│
└── data/
    ├── input.csv                  # outputs of the scrappers
    ├── preprocessed.parquet       # outputs of nlp_prep.py
    └── semantic.parquet           # outputs of nlp_semantic.py
```
