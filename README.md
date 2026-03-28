# NLP_pipeline

project/
│
├── nlp_prep.py                    # main preprocessing pipeline
├── semantic.py                    # sentiment pipeline
│
├── config.py                      # all constants & dictionaries
├── text_utils.py                  # low-level helpers (lemma, normalization)
├── text_features.py               # tagging, topics, detection logic
│
└── data/
    ├── input.csv                  # outputs of the scrappers
    ├── preprocessed.parquet       # outputs of nlp_prep.py
    └── semantic.parquet           # outputs of the semantic.py
