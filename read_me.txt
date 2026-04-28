Structure

project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── predictions/
│
├── models/
│
├── scripts/
│   ├── preprocess_data.py
│   ├── train_models.py
│   └── predict.py
│
├── src/
│   ├── config.py
│   │
│   ├── ml/
│   │   ├── feature_builder.py
│   │   ├── predictor.py
│   │   └── trainer.py
│   │
│   ├── nlp/
│   │   ├── cleaner.py
│   │   ├── feature_extractor.py
│   │   ├── lemmatizer.py
│   │   ├── sentiment_analyzer.py
│   │   ├── topic_detector.py
│   │   └── pipeline.py
│   │
│   └── dashboard.py

Step 1: Scrappers
Step 2: Preprocess 
run "python scripts/preprocess_data.py"
results in data/processed/comments_enriched.parquet
Step 3: Train models
run "python scripts/train_models.py"
results in models/
    feature_builder.pkl
    cluster_model.pkl
    sentiment_model.pkl
    cluster_encoder.pkl
    sentiment_encoder.pkl
Step 4: Predictions
run "python scripts/predict.py"
results in data/predictions/*.parquet
Step 5: Dashboard
run "streamlit run src/dashboard.py"