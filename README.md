# Full Prediction Process Project

This project provides a comprehensive pipeline for natural language processing tasks, including text preprocessing, cluster classification, and sentiment analysis. It leverages both traditional machine learning models (Logistic Regression, Linear SVM) and a transformer-based model for sentiment prediction.

## Features

-   **Data Preprocessing:** Cleans and lemmatizes text data, and applies regex-based classification for initial clustering.
-   **ML Model Training:** Trains and evaluates Logistic Regression and Linear SVM models for both cluster and sentiment prediction on pre-clustered datasets.
-   **Hyperparameter Tuning:** Includes functionality for hyperparameter optimization to find the best model configurations.
-   **Transformer-based Sentiment Analysis:** Integrates a pre-trained transformer model for advanced sentiment prediction.
-   **Prediction on New Data:** Capable of processing new, unclustered datasets to predict both clusters and sentiment.

## Quick Start

To get this project up and running, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd full_prediction_project
    ```

2.  **Install dependencies:**

    Ensure you have Python 3.8+ installed. Then, install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare your data:**

    Place your input CSV file (e.g., `input_data.csv`) containing a column named `Комментарий` (for text) and optionally `cluster_mode` and `sentiment_mode` for training/evaluation in the project root or specify its path.

4.  **Run the full prediction pipeline:**

    ```bash
    python main.py --input_file path/to/your/dataset.csv
    ```

    *(More detailed command-line arguments and usage will be provided in `main.py` and its documentation.)*

## Project Structure

-   `README.md`: This file.
-   `requirements.txt`: Lists all Python dependencies.
-   `src/`:
    -   `preprocessing.py`: Contains functions for text cleaning, lemmatization, and regex-based initial clustering.
    -   `sentiment_model.py`: Implements the transformer-based sentiment analysis using `blanchefort/rubert-base-cased-sentiment`.
    -   `ml_models.py`: Handles TF-IDF vectorization, data splitting, hyperparameter tuning, and training/evaluation of Logistic Regression and Linear SVM for cluster and sentiment prediction.
    -   `predict.py`: Orchestrates the prediction process for new, unclustered data.
-   `main.py`: The entry point for running the entire pipeline, managing data flow between modules.

## Usage Examples

*(This section will be expanded with specific examples once the core scripts are developed.)*

## Configuration

*(Details on how to configure models, thresholds, or other parameters will be added here.)*

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.
