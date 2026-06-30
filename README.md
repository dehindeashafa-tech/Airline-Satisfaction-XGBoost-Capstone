# Airline-Satisfaction-XGBoost-Capstone
Advanced Predictive Modeling for Airline Satisfaction – XGBoost Deployment (Discovery-to-Action Strategy)
# Customer Satisfaction Prediction for Airline

## Overview

This project aims to predict customer satisfaction for an airline based on various service touchpoints. We explore several machine learning models, including Decision Trees, Logistic Regression, Random Forest, and XGBoost, to identify the most effective model for deployment. The goal is to provide actionable insights to airline leadership to improve customer experience and retention.

## Project Workflow

The analysis follows a structured Machine Learning workflow:

1.  **Data Loading and Preparation:** The `invistico_Airline1.csv` dataset is loaded, unnecessary columns are dropped, and missing values in 'Arrival Delay in Minutes' are imputed with the mean. Categorical features like 'Customer Type', 'Type of Travel', 'Class', and the target variable 'satisfaction' are label encoded.
2.  **Data Splitting:** The preprocessed data is split into training and testing sets (70% train, 30% test) using `train_test_split` with stratification to maintain class distribution.
3.  **Model Development & Tuning:**
    *   **Decision Tree Classifier:** Tuned using `GridSearchCV` to find optimal `max_depth`, `min_samples_split`, and `min_samples_leaf` based on `f1_weighted` scoring.
    *   **Logistic Regression:** Trained on scaled data with `StandardScaler`.
    *   **XGBoost Classifier:** Initial model trained, followed by hyperparameter tuning using `GridSearchCV` for `max_depth`, `learning_rate`, and `n_estimators`.
    *   **Random Forest Classifier:** Trained with default parameters.
4.  **Model Evaluation:** Each model's performance is evaluated using confusion matrices, F1-score for the 'Satisfied' class, and a comprehensive classification report.
5.  **Comparative Analysis:** A comparative table is generated, summarizing Accuracy, Weighted Precision, Weighted Recall, and Weighted F1-score for all models. Confusion matrices are also compared side-by-side.
6.  **Feature Importance:** For the best-performing model (Tuned XGBoost), feature importance is extracted and visualized to identify key customer touchpoints influencing satisfaction.
7.  **Executive Summary & Recommendations:** A concise executive summary outlines actionable strategies for airline leadership based on model insights.
8.  **Limitations & Deployment Considerations:** Discussion of model limitations, deployment strategies, and next steps for production monitoring.

## Final Model Selection Rationale

After comprehensive evaluation, the **Tuned XGBoost Classifier** is recommended for deployment. This decision is based on its superior performance across key metrics, particularly its ability to minimize false negatives (predicting a dissatisfied customer as satisfied).

**Key Justifications:**

*   **Highest Overall Performance:** Tuned XGBoost consistently achieved the highest Accuracy, Weighted Precision, Weighted Recall, and Weighted F1-score compared to Decision Tree, Logistic Regression, and Random Forest.
*   **Minimizing False Negatives:** For an airline, correctly identifying dissatisfied customers is paramount to prevent churn and negative feedback. The Tuned XGBoost model exhibited the lowest number of false negatives, making it highly effective for proactive intervention.
*   **Balanced Performance:** The F1-score for the 'Satisfied' class was highest for Tuned XGBoost, indicating a strong balance between precision and recall for both customer satisfaction categories.

## Key Business Insights & Actionable Strategies

The XGBoost feature importance analysis revealed critical customer touchpoints:

1.  **Inflight Entertainment:** Top predictor. **Action:** Invest in upgrading systems and content variety.
2.  **Seat Comfort:** Fundamental aspect. **Action:** Conduct surveys, explore seat upgrades and legroom improvements.
3.  **Online Support:** Efficiency is key. **Action:** Streamline channels, reduce response times, and enhance chatbot capabilities.
4.  **Customer Type:** Different expectations for Business vs. Personal Travel. **Action:** Tailor services (e.g., Wi-Fi for business, family entertainment for personal).
5.  **Ease of Online Booking:** First impression matters. **Action:** Refine the booking interface for intuition and speed.
6.  **Class (Business, Eco, etc.):** Service levels influence satisfaction. **Action:** Review and enhance unique offerings per class to exceed expectations.

## Setup and Usage

To run this notebook and reproduce the analysis, please follow these steps:

1.  **Environment:** This notebook is designed to run in a Google Colab environment.
2.  **Dependencies:** Ensure you have the following libraries installed. They can be installed using `pip`:
    ```bash
    pip install pandas scikit-learn matplotlib seaborn xgboost
    ```
    (Most of these are pre-installed in Colab).
3.  **Data:** Upload the `invistico_Airline1.csv` file to your Colab environment or ensure it's accessible at `/content/invistico_Airline1.csv`.
4.  **Execution:** Run all cells sequentially within the notebook. The output will guide you through the data preparation, model training, evaluation, and insights.

## Code Documentation

The notebook includes extensive Markdown cells explaining each step of the process, tuning strategies, comparative analyses, and the interpretation of results and recommendations for leadership.

## Next Steps for Production

*   **Continuous Monitoring:** Implement systems to monitor model performance and data/concept drift.
*   **Regular Retraining:** Establish a schedule for retraining the model with fresh data.
*   **A/B Testing:** Evaluate the impact of implemented strategies through A/B testing.
*   **API Development:** Create an API for seamless integration with airline operational systems.
