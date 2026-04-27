import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import numpy as np
import json

RANDOM_STATE = 0

def train_and_evaluate_ml_models(df, text_column, cluster_target_column, sentiment_target_column):
    results = pd.DataFrame(columns=['analysed_parameter', 'alg', 'hyperparametes', 'metric_name', 'score_train', 'score_valid'])

    # Prepare data for cluster prediction
    cluster_texts = df[text_column].fillna("")
    le_cluster = LabelEncoder()
    cluster_target = le_cluster.fit_transform(df[cluster_target_column].replace('none', 'no_cluster'))

    # Prepare data for sentiment prediction (ML-based)
    sentiment_texts = df[text_column].fillna("")
    le_sentiment = LabelEncoder()
    sentiment_target = le_sentiment.fit_transform(df[sentiment_target_column])

    # Split data for clusters
    Xc_train_all, Xc_test, c_train_all, c_test = train_test_split(cluster_texts, cluster_target, test_size=0.3, random_state=RANDOM_STATE, stratify=cluster_target)
    Xc_train, Xc_valid, c_train, c_valid = train_test_split(Xc_train_all, c_train_all, test_size=0.5, random_state=RANDOM_STATE, stratify=c_train_all)

    # Split data for sentiment
    Xs_train_all, Xs_test, s_train_all, s_test = train_test_split(sentiment_texts, sentiment_target, test_size=0.3, random_state=RANDOM_STATE, stratify=sentiment_target)
    Xs_train, Xs_valid, s_train, s_valid = train_test_split(Xs_train_all, s_train_all, test_size=0.5, random_state=RANDOM_STATE, stratify=s_train_all)

    # TF-IDF Vectorization for clusters
    count_idf_cluster = TfidfVectorizer()
    X_cluster_train = count_idf_cluster.fit_transform(Xc_train)
    X_cluster_valid = count_idf_cluster.transform(Xc_valid)
    X_cluster_test = count_idf_cluster.transform(Xc_test)

    # TF-IDF Vectorization for sentiment
    count_idf_sentiment = TfidfVectorizer()
    X_sentiment_train = count_idf_sentiment.fit_transform(Xs_train)
    X_sentiment_valid = count_idf_sentiment.transform(Xs_valid)
    X_sentiment_test = count_idf_sentiment.transform(Xs_test)

    # --- Logistic Regression for Clusters ---
    print("Training Logistic Regression for Clusters...")
    clf_lr_cluster = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    parameters_lr = parameters = [
            {'solver': ['lbfgs'], 'penalty': ['l2'], 'C': [0.5, 1.0, 1.5]},
            {'solver': ['saga'], 'penalty': ['l1', 'l2'], 'C': [0.5, 1.0], 'fit_intercept': [True]}
        ]
    rnd_lr_cluster = RandomizedSearchCV(clf_lr_cluster,
                                        parameters,
                                        n_iter=10,  # ключевой параметр
                                        scoring='f1_macro',
                                        n_jobs=-1,  # параллелизация
                                        random_state=RANDOM_STATE
                                    )
    rnd_lr_cluster.fit(X_cluster_train, c_train)
    best_lr_cluster = rnd_lr_cluster.best_estimator_
    
    y_pred_train_lr_cluster = best_lr_cluster.predict(X_cluster_train)
    y_pred_valid_lr_cluster = best_lr_cluster.predict(X_cluster_valid)
    score_train_lr_cluster = f1_score(c_train, y_pred_train_lr_cluster, average='macro')
    score_valid_lr_cluster = f1_score(c_valid, y_pred_valid_lr_cluster, average='macro')
    results = pd.concat([results, pd.DataFrame([['cluster', 'Logistic Regression', json.dumps(rnd_lr_cluster.best_params_), 'f1 macro', score_train_lr_cluster, score_valid_lr_cluster]], columns=results.columns)], ignore_index=True)
    
    # Per-cluster F1-score for Logistic Regression on Clusters
    print("--- Classification Report for Logistic Regression (Clusters - Test Set) ---")
    y_pred_lr_cluster_test = best_lr_cluster.predict(X_cluster_test)
    print(classification_report(c_test, y_pred_lr_cluster_test, target_names=le_cluster.inverse_transform(np.unique(c_test)), zero_division=0))

    # --- Linear SVM for Clusters ---
    print("Training Linear SVM for Clusters...")
    clf_svm_cluster = LinearSVC(random_state=RANDOM_STATE, max_iter=2000, tol=0.001)
    parameters_svm = parameters = [
                                    # стандартный быстрый вариант
                                    {
                                        'penalty': ['l2'],
                                        'loss': ['hinge', 'squared_hinge'],
                                        'C': [0.25, 0.5],
                                        'fit_intercept': [True]
                                    },
                                    
                                    # l1 работает только так
                                    {
                                        'penalty': ['l1'],
                                        'loss': ['squared_hinge'],
                                        'dual': [False],
                                        'C': [0.25, 0.5],
                                        'fit_intercept': [True]
                                    }
                                ]
    rnd_svm_cluster = rnd = RandomizedSearchCV(
                                                clf_svm_cluster,
                                                parameters,
                                                n_iter=6,   # было 10 → можно меньше
                                                scoring='f1_macro',
                                                n_jobs=-1,
                                                random_state=RANDOM_STATE
                                            )
    rnd_svm_cluster.fit(X_cluster_train, c_train)
    best_svm_cluster = rnd_svm_cluster.best_estimator_
    
    y_pred_train_svm_cluster = best_svm_cluster.predict(X_cluster_train)
    y_pred_valid_svm_cluster = best_svm_cluster.predict(X_cluster_valid)
    score_train_svm_cluster = f1_score(c_train, y_pred_train_svm_cluster, average='macro')
    score_valid_svm_cluster = f1_score(c_valid, y_pred_valid_svm_cluster, average='macro')
    results = pd.concat([results, pd.DataFrame([['cluster', 'Linear SVM', json.dumps(rnd_svm_cluster.best_params_), 'f1 macro', score_train_svm_cluster, score_valid_svm_cluster]], columns=results.columns)], ignore_index=True)

    # Per-cluster F1-score for Linear SVM on Clusters
    print("--- Classification Report for Linear SVM (Clusters - Test Set) ---")
    y_pred_svm_cluster_test = best_svm_cluster.predict(X_cluster_test)
    print(classification_report(c_test, y_pred_svm_cluster_test, target_names=le_cluster.inverse_transform(np.unique(c_test)), zero_division=0))

    # --- Logistic Regression for Sentiment ---
    print("Training Logistic Regression for Sentiment...")
    clf_lr_sentiment = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)
    rnd_lr_sentiment = RandomizedSearchCV(clf_lr_sentiment, parameters_lr, scoring='f1_macro', error_score=np.nan, random_state=RANDOM_STATE)
    rnd_lr_sentiment.fit(X_sentiment_train, s_train)
    best_lr_sentiment = rnd_lr_sentiment.best_estimator_
    
    y_pred_train_lr_sentiment = best_lr_sentiment.predict(X_sentiment_train)
    y_pred_valid_lr_sentiment = best_lr_sentiment.predict(X_sentiment_valid)
    score_train_lr_sentiment = f1_score(s_train, y_pred_train_lr_sentiment, average='macro')
    score_valid_lr_sentiment = f1_score(s_valid, y_pred_valid_lr_sentiment, average='macro')
    results = pd.concat([results, pd.DataFrame([['sentiment', 'Logistic Regression', json.dumps(rnd_lr_sentiment.best_params_), 'f1 macro', score_train_lr_sentiment, score_valid_lr_sentiment]], columns=results.columns)], ignore_index=True)

    # Per-sentiment F1-score for Logistic Regression on Sentiment
    print("--- Classification Report for Logistic Regression (Sentiment - Test Set) ---")
    y_pred_lr_sentiment_test = best_lr_sentiment.predict(X_sentiment_test)
    print(classification_report(s_test, y_pred_lr_sentiment_test, target_names=le_sentiment.inverse_transform(np.unique(s_test)), zero_division=0))

    # --- Linear SVM for Sentiment ---
    print("Training Linear SVM for Sentiment...")
    clf_svm_sentiment = LinearSVC(random_state=RANDOM_STATE, max_iter=2000, tol=0.001)
    rnd_svm_sentiment = RandomizedSearchCV(clf_svm_sentiment, parameters_svm, scoring='f1_macro', error_score=np.nan, random_state=RANDOM_STATE)
    rnd_svm_sentiment.fit(X_sentiment_train, s_train)
    best_svm_sentiment = rnd_svm_sentiment.best_estimator_
    
    y_pred_train_svm_sentiment = best_svm_sentiment.predict(X_sentiment_train)
    y_pred_valid_svm_sentiment = best_svm_sentiment.predict(X_sentiment_valid)
    score_train_svm_sentiment = f1_score(s_train, y_pred_train_svm_sentiment, average='macro')
    score_valid_svm_sentiment = f1_score(s_valid, y_pred_valid_svm_sentiment, average='macro')
    results = pd.concat([results, pd.DataFrame([['sentiment', 'Linear SVM', json.dumps(rnd_svm_sentiment.best_params_), 'f1 macro', score_train_svm_sentiment, score_valid_svm_sentiment]], columns=results.columns)], ignore_index=True)

    # Per-sentiment F1-score for Linear SVM on Sentiment
    print("--- Classification Report for Linear SVM (Sentiment - Test Set) ---")
    y_pred_svm_sentiment_test = best_svm_sentiment.predict(X_sentiment_test)
    print(classification_report(s_test, y_pred_svm_sentiment_test, target_names=le_sentiment.inverse_transform(np.unique(s_test)), zero_division=0))

    print("--- Model Training and Evaluation Complete ---")
    print(results)
    
    return {
        'cluster_vectorizer': count_idf_cluster, 'sentiment_vectorizer': count_idf_sentiment,
        'cluster_label_encoder': le_cluster, 'sentiment_label_encoder': le_sentiment,
        'best_lr_cluster': best_lr_cluster, 'best_svm_cluster': best_svm_cluster,
        'best_lr_sentiment': best_lr_sentiment, 'best_svm_sentiment': best_svm_sentiment,
        'X_cluster_test': X_cluster_test, 'c_test': c_test,
        'X_sentiment_test': X_sentiment_test, 's_test': s_test
    }

def evaluate_final_models(trained_models):
    print("--- Final Model Evaluation on Test Sets ---")
    # Evaluate best cluster model
    best_cluster_model = trained_models['best_lr_cluster'] # Assuming LR is chosen, can be improved to select dynamically
    if trained_models['best_svm_cluster'].score(trained_models['X_cluster_test'], trained_models['c_test']) > best_cluster_model.score(trained_models['X_cluster_test'], trained_models['c_test']):
        best_cluster_model = trained_models['best_svm_cluster']
    
    y_pred_cluster_test = best_cluster_model.predict(trained_models['X_cluster_test'])
    f1_cluster_test = f1_score(trained_models['c_test'], y_pred_cluster_test, average='macro')
    print(f"Best Cluster Model (via {best_cluster_model.__class__.__name__}) F1-score on test set: {f1_cluster_test:.4f}")

    # Evaluate best sentiment ML model
    best_sentiment_model = trained_models['best_lr_sentiment'] # Assuming LR is chosen, can be improved to select dynamically
    if trained_models['best_svm_sentiment'].score(trained_models['X_sentiment_test'], trained_models['s_test']) > best_sentiment_model.score(trained_models['X_sentiment_test'], trained_models['s_test']):
        best_sentiment_model = trained_models['best_svm_sentiment']

    y_pred_sentiment_test = best_sentiment_model.predict(trained_models['X_sentiment_test'])
    f1_sentiment_test = f1_score(trained_models['s_test'], y_pred_sentiment_test, average='macro')
    print(f"Best ML Sentiment Model (via {best_sentiment_model.__class__.__name__}) F1-score on test set: {f1_sentiment_test:.4f}")

    return {'best_cluster_model': best_cluster_model, 'best_sentiment_model': best_sentiment_model}
