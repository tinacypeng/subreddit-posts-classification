## Tina's Project - Subreddit Posts Classification (Web APIs & NLP Application)

### Description

The purpose of this project is to build classification models based on data collected from two subreddits and compare model performances on classifying a Reddit post to either of the two subreddits.

For this project, I choose Reddit posts from two sport subreddits, [`r/nba`](https://www.reddit.com/r/nba/) and [`r/nfl`](https://www.reddit.com/r/nfl/), as my basic training datasets. And try to make the best prediction through a classification model of whether this new Reddit post is from `r/nba` or `r/nfl`.

---
### Data Collection

In data collection process, I use [Pushshift's](https://github.com/pushshift/api) API to collect a total of 3,997 post **titles** and **contents** from subreddit [`r/nba`](https://www.reddit.com/r/nba/) and [`r/nfl`](https://www.reddit.com/r/nfl/).

* `r/nba` : 
    * Duration of post date : 2022/10/18 - 2022/10/23
    * Number of posts : 1,997
* `r/nfl` :
    * Duration of post date : 2022/10/16 - 2022/10/23
    * Number of posts : 2,000

```
* code: 01_Data_Collection.ipynb
* data output: df_nba.csv, df_nfl.csv
```
---
### Data Cleaning and EDA

#### Data Cleaning

1. Combine two datasets (df_nba, df_nfl) into one large dataset.
2. Check all missing values and fill them accordingly.
3. Remove non-English alphabets to avoid adding some spam-likely data into model training step.
4. Convert target variable (`subreddit`) into binary labels.

#### EDA

1. Compare distribution of post's title lengths between Reddit posts from NBA and NFL.
2. Sentiment analysis of all titles.
3. Output the cleaned dataset in the data file to be used in the next step.

```
* code: 02_Data_Cleaning_and_EDA.ipynb
* data input: df_nba.csv, df_nfl.csv
* data output: df_cleaned.csv
```
---
### Data Preprocessing
After cleaning the datasets, I can now take a deeper look on which words are the top common words in the whole datasets, the NBA channel, and the NFL channel.

* Using `CountVectorizer` with 1-gram and 2-gram to find top 10 common 1-word and 2-word in each subdatasets (NBA and NFL with positive or negative sentiment).

```
* code: 03_Data_Preprocessing.ipynb
* data input: df_cleaned.csv
* data output: df_cvec_1.csv, df_cvec_2.csv
```
---
### Modeling - MNB, Logistic Regression, KNN
In the first modeling process, I try 3 classifier machine learning models (Multinomial Naive Bayes, Logistic Regression, and KNN) with 2 vectorizer (Count and TF-IDF).

1. Model 1: (Count Vectorizer + Multinomial Naive Bayes)
 * training score: 0.956
 * testing score: 0.930
 * cross-validation score: 0.917
 * accuracy rate: 0.923
    
2. Model 2: (TF-IDF Vectorizer + Multinomial Naive Bayes)
 * training score: 0.970
 * testing score: 0.930
 * cross-validation score: 0.919
 * accuracy rate: 0.925
 
3. Model 3: (TF-IDF Vectorizer + Logistic Regression)
 * training score: 0.978
 * testing score: 0.930
 * cross-validation score: 0.915
 * accuracy rate: 0.916
    
4. Model 4: (TF-IDF Vectorizer + KNN)
 * training score: 0.911
 * testing score: 0.900
 * cross-validation score: 0.917
 * accuracy rate: 0.888

```
* code: 04_Modeling_MNB_LogReg_KNN.ipynb
* data input: df_cleaned.csv
```
---
### Modeling - Boosting
After learning boosting, I go back and train another model by using boosting and try to make a better prediction.

However, compared to Multinomial Naive Bayes, Gradient Boosting Classifier doesn't get higher scores on either a 1-word or 2-word count vector. The highest scores I got from Gradient Boosting Classifier is only 0.848 on training and 0.854 on testing.

```
* code: 04_Modeling_Boosting.ipynb
* data input: df_cvec_1.csv, df_cvec_2.csv
```
---
### Summary
* **Summary table**:

|  | CVEC | TF-IDF | MNB | KNN | LogReg | Avg. Score | Rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model 1 | <font color='dodgerblue'>v</font> | <font color='gray'>x</font> | <font color='dodgerblue'>v</font> | <font color='gray'>x</font> | <font color='gray'>x</font> | 0.917 | 2 |
| Model 2 | <font color='gray'>x</font> | <font color='dodgerblue'>v</font> | <font color='dodgerblue'>v</font> | <font color='gray'>x</font> | <font color='gray'>x</font> | 0.919 | <font color='red'>1</font> |
| Model 3 | <font color='gray'>x</font> | <font color='dodgerblue'>v</font> | <font color='gray'>x</font> | <font color='gray'>x</font> | <font color='dodgerblue'>v</font> | 0.915 | 4 |
| Model 4 | <font color='gray'>x</font> | <font color='dodgerblue'>v</font> | <font color='gray'>x</font> | <font color='dodgerblue'>v</font> | <font color='gray'>x</font> | 0.917 | 2 |

* Overall, all 4 models I train in the first modeling process have similar average scores at around 91.7%, which is good! And model 2, composed of **Multinomial Naive Bayes** with **TF-IDF Vectorizer**, got the highest average score at 91.9%. 

* However, if we care more about the overfitting problem, model 4 (**KNN + TF-IDF Vectorizer**) might be the best choice since its training and testing scores are the least different.

* Common word analysis is the most interesting and important part of NLP. For some special terms in specific fields, after we go through the common word analysis part, it can automatically help us separate the special terms in different fields. For example, in this project, I have basically not enough knowledge of either the NBA or the NFL. But, after common word analysis, I learned some basketball terms and get a better understanding of these two sports. And I found it really interesting!! 

---
## Directory Structure
```
subreddit-posts-classification
|__ code
|   |__ 01_Data_Collection.ipynb 
|   |__ 02_Data_Cleaning_and_EDA.ipynb   
|   |__ 03_Data_Preprocessing.ipynb
|   |__ 04_Modeling_1_MNB_LogReg_KNN.ipynb 
|   |__ 04_Modeling_2_Boosting.ipynb  
|
|__ data
|   |__ df_cleaned.csv
|   |__ df_cvec_1.csv
|   |__ df_cvec_2.csv
|   |__ df_nba.csv
|   |__ df_nfl.csv
|
|__ images
|   |__ commonwords_1_all.png
|   |__ commonwords_1_nba.png
|   |__ commonwords_1_nfl.png
|   |__ commonwords_2_all.png
|   |__ commonwords_2_nba.png
|   |__ commonwords_2_nba_neg.png
|   |__ commonwords_2_nba_pos.png
|   |__ commonwords_2_nfl.png
|   |__ commonwords_2_nfl_neg.png
|   |__ commonwords_2_nfl_pos.png
|   |__ tinap.png
|   |__ title_wordcount_all.png
|   |__ title_wordcount_nba.png
|   |__ title_wordcount_nfl.png
|
|__ subreddit_posts_classification_presentation_slides.pdf
|__ README.md
```