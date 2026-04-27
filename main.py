import pandas as pd
import argparse
import os
import pickle
from src.preprocessing import preprocess_data
from src.ml_models import train_and_evaluate_ml_models, evaluate_final_models
from src.predict import predict_on_new_data, load_model_artifacts

def main(input_file, output_file, train_models=False, model_dir="models"):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    # Ensure the 'new comment' column exists for preprocessing
    if 'Комментарий' not in df.columns:
        raise ValueError("Input CSV must contain a 'Комментарий' column for text data.")

    df_processed = preprocess_data(df.copy())
    print("Data preprocessing complete.")

    trained_artifacts = {}
    if train_models:
        print("--- Training and Evaluating ML Models ---")
        if 'cluster_mode' not in df.columns or 'sentiment_mode' not in df.columns:
            print("Warning: 'cluster_mode' or 'sentiment_mode' not found. Skipping ML model training and evaluation.")
        else:
            # Filter out rows where target columns are NaN before training
            df_trainable = df_processed.dropna(subset=['cluster_mode', 'sentiment_mode']).copy()
            if df_trainable.empty:
                print("No valid data for training after dropping NaNs in target columns. Skipping ML model training.")
            else:
                trained_artifacts = train_and_evaluate_ml_models(
                    df_trainable, 'new comment', 'cluster_mode', 'sentiment_mode'
                )
                final_models = evaluate_final_models(trained_artifacts)
                trained_artifacts.update(final_models)

                # Save model artifacts
                os.makedirs(model_dir, exist_ok=True)
                for key, value in trained_artifacts.items():
                    if key not in ['X_cluster_test', 'c_test', 'X_sentiment_test', 's_test']:
                        with open(os.path.join(model_dir, f'{key}.pkl'), 'wb') as f:
                            pickle.dump(value, f)
                print(f"Trained ML models and vectorizers saved to {model_dir}/")
    else:
        print(f"--- Loading pre-trained models from {model_dir}/ for prediction ---")
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            print(f"Error: Model directory {model_dir} is empty or does not exist. Cannot proceed with prediction without trained models. Please run with --train_models first.")
            return
        try:
            trained_artifacts = load_model_artifacts(model_dir)
        except FileNotFoundError as e:
            print(f"Error loading model artifacts: {e}. Ensure all model files exist in {model_dir}/")
            return

    print("--- Making Predictions ---")
    if trained_artifacts:
        df_result = predict_on_new_data(df_processed.copy(), trained_artifacts)
        print("Predictions complete.")

        print(f"Saving results to {output_file}...")
        df_result.to_csv(output_file, index=False)
        print("Process finished successfully!")
    else:
        print("No models available for prediction. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full NLP Prediction Process.")
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input CSV file containing comments.')
    parser.add_argument('--output_file', type=str, default='predictions.csv',
                        help='Path to save the output CSV file with predictions.')
    parser.add_argument('--train_models', action='store_true',
                        help='Set this flag to train ML models and save them. Requires "cluster_mode" and "sentiment_mode" columns in input_file.')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save/load trained ML models and vectorizers.')

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.train_models, args.model_dir)
