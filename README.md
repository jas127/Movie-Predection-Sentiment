# Movie Success Prediction and Sentiment Study

This project aims to **predict movie success (specifically, box office revenue)** and analyze the **sentiment expressed in movie titles and viewer reviews**. It combines methodologies for regression modeling and sentiment analysis using Python and various datasets sourced from IMDb.

## Project Objectives

The core objectives of this project are:
1. **Predict movie box office success** using historical data and movie metadata.
2. **Analyze sentiment** found in movie titles across different genres.
3. **Classify sentiment** of user-written movie reviews.

## Dataset

The project utilizes data derived from IMDb:

- **Box Office Data:** A CSV file (`box_office_data.csv`) containing top movies by worldwide gross. Key fields include movie title, worldwide revenue, release year, genres, MPAA rating, IMDb vote count, original language, and production countries. This data is used for both revenue prediction and title sentiment analysis.
- **Movie Review Data:** A dataset containing 50,000 labeled movie reviews, split evenly between positive and negative sentiment. Each entry includes the review text and its corresponding sentiment label (pos or neg).

## Methodology

### Movie Box Office Revenue Prediction

- **Data preprocessing:** Drop rows missing the target variable (worldwide revenue) and handle other missing values.
- **Features used:** Year, genres, rating, vote count, original language, production countries; target is worldwide revenue.
- **Pipeline construction:**
  - Numerical features (Year, vote count) → StandardScaler
  - Categorical features (genres, rating, language, countries) → OneHotEncoder
  - Model: RandomForestRegressor
- **Train/test split:** 80% training, 20% testing.
- **Training & prediction:** Fit the pipeline on training data and predict on the test set.

### Movie Title Sentiment Analysis

- **Data cleaning:** Remove entries missing title or genre.
- **Genre explosion:** Split multi-genre entries so each row represents a single title–genre pair.
- **Sentiment scoring:** Compute a polarity score for each title on a –1 (negative) to +1 (positive) scale.
- **Aggregation:** Calculate the average title polarity for each genre.

### Movie Review Sentiment Analysis

- **Preprocessing:** Lowercase conversion, removal of HTML tags, punctuation, and extra whitespace.
- **Sentiment scoring:** Use VADER to compute positive, neutral, negative, and compound scores for each review.
- **Label assignment:** If compound score ≥ 0 → positive; otherwise negative.
- **Evaluation:** Accuracy, precision, recall, F1-score, and confusion matrix.

## Key Findings & Results

- **Revenue Prediction:**  
  - R² score of approximately 0.528, explaining about 52.8% of variance in worldwide gross.  
  - Mean Squared Error reflects high variance in box office figures.  
  - Primary predictors include vote count, release year, and certain genre indicators.

- **Title Sentiment:**  
  - Highest average title positivity in Family and Animation genres.  
  - Lowest average title positivity in Horror and Thriller genres.  
  - Comedy and Action titles show wide sentiment ranges; Adventure and Sci-Fi cluster near neutral.

- **Review Sentiment:**  
  - Overall accuracy around 85%.  
  - Positive reviews tend to score around +0.8; negative reviews cluster near –0.5.  
  - Slight bias toward false negatives suggests threshold tuning could improve balance.

## Project Structure

. ├── movie-revenue-prediction/ ├── review-sentiment-analysis/ ├── README.md ├── movie-genre.pdf ├── movie-revenue.pdf ├── movie-review.pdf └── project-report.pdf

*   `movie-revenue-prediction/`: Contains code and notebooks related to box office revenue prediction.
*   `review-sentiment-analysis/`: Contains code and notebooks related to movie review sentiment analysis.
*   `README.md`: This file.
*   `.pdf` files: Source documents and the combined project report.

## Requirements & Tools

- Python  
- pandas  
- scikit-learn  
- nltk (for VADER)  
- matplotlib  

## Limitations

The projects have several limitations:

* **Sentiment Analysis (General):** Lexicon-based approaches like VADER may struggle with context, sarcasm, domain-specific language, or interpreting proper nouns. The simple threshold rule for review classification is basic.  
* **Title Sentiment:** The analysis focuses only on the title text, ignoring full reviews or plot details. Multi-genre films can dilute sentiment signals.  
* **Revenue Prediction:** The model uses a limited set of features and doesn’t account for factors like marketing budget, star power, or seasonal releases. Simple one-hot encoding might not handle rare categories well. Extreme box office outliers can skew results.  
* **Review Sentiment:** The evaluation was performed on a balanced dataset; performance might differ on skewed real-world data.

## Future Work

Potential areas for future development include:

* Adding more relevant features (e.g., budget, cast, director) and performing hyperparameter tuning or using alternative models (Gradient Boosting, Neural Networks) for revenue prediction.  
* Performing keyword frequency analysis, applying advanced sentiment models (e.g., transformers), or exploring temporal trends for title sentiment.  
* Optimizing the sentiment classification threshold, using ensemble methods, or employing deep learning approaches (e.g., fine-tuning BERT) for review sentiment. Conducting error analysis to identify specific patterns in misclassifications.

## References

* Scikit-Learn Documentation  
* IMDb Datasets  
* Hutto, C.J. & Gilbert, E. “VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text”  
