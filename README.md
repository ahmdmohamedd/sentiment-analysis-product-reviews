# Sentiment Analysis on Product Reviews

## Overview

This project performs sentiment analysis on product reviews to classify them as either **positive** or **negative**. The analysis is conducted using a **Naive Bayes classifier**, a popular machine learning model for text classification tasks. The system processes raw text reviews from the IMDb dataset and makes sentiment predictions by leveraging natural language processing (NLP) techniques such as text preprocessing, tokenization, and feature extraction.

## Project Structure

```
sentiment-analysis-product-reviews/
│
├── sentiment_analysis_product_reviews.ipynb  # Jupyter Notebook with code for sentiment analysis
├── README.md                                # This readme file
```

## Prerequisites

To run this project, ensure that you have the following installed:

- Python (>= 3.7)
- Jupyter Notebook
- Libraries: 
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `nltk`

## Getting Started

1. **Clone this repository** to your local machine:

   ```bash
   git clone https://github.com/ahmdmohamedd/sentiment-analysis-product-reviews.git
   ```

2. **Navigate** to the project folder:

   ```bash
   cd sentiment-analysis-product-reviews
   ```

3. **Open the Jupyter Notebook**:

   ```bash
   jupyter notebook sentiment_analysis_product_reviews.ipynb
   ```

4. **Run the notebook** cell by cell to perform sentiment analysis on the dataset.

## How It Works

1. **Data Loading**:  
   The project uses a publicly available IMDb reviews dataset. This dataset contains movie reviews labeled as either "positive" or "negative". We load the dataset using Pandas.

2. **Text Preprocessing**:  
   Text data is cleaned by removing stopwords, special characters, and performing tokenization. This step ensures that the data is in a format suitable for machine learning models.

3. **Feature Extraction**:  
   The **Bag of Words** model is used to convert the cleaned text data into numerical features. The `CountVectorizer` from `scikit-learn` is used to create this transformation.

4. **Model Training**:  
   The **Multinomial Naive Bayes classifier** is used to train the model on the preprocessed dataset. It is particularly well-suited for text classification tasks with discrete features.

5. **Model Evaluation**:  
   The trained model is evaluated using various metrics such as accuracy, precision, recall, and the confusion matrix to assess its performance on the test data.

6. **Prediction**:  
   The system predicts the sentiment of new reviews using the trained model, outputting whether the review is "positive" or "negative."

## Example Usage

Once the system is running, you can predict the sentiment of new product reviews. Here’s an example:

```python
new_review = "This product is amazing and works great!"
new_review_processed = preprocess_review(new_review)
new_review_vectorized = vectorizer.transform([new_review_processed])
prediction = model.predict(new_review_vectorized)

print("Sentiment:", "Positive" if prediction == 1 else "Negative")
```

## Evaluation

The system evaluates the model's performance using:

- **Accuracy**: Percentage of correct predictions
- **Confusion Matrix**: Helps visualize true positives, true negatives, false positives, and false negatives
- **Classification Report**: Provides precision, recall, and F1-score for each class

### Example Results:

```
Accuracy: 85.81%
Precision: 0.86, Recall: 0.86, F1-Score: 0.86
```

## Future Enhancements

- **Hyperparameter Tuning**: Experiment with different machine learning algorithms and hyperparameters to improve accuracy.
- **Advanced Text Processing**: Implement more advanced text preprocessing techniques, such as stemming and lemmatization.
- **Deployment**: Package the model into a web application for real-time sentiment analysis.
