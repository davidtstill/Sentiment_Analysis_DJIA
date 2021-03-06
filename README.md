# Columbia Fintech Bootcamp - Project 2
*Assignment - Use Jupyter, Google Colab, or AWS SageMaker to prepare a training and testing dataset and train a machine-learning model.*

# Analyzing Daily News Headlines to Predict the Direction of the Dow Jones Industrial Average

## Team Members:
- David Still
- Charles Xia
- Ian Walter

# Topic of the Analysis
Can we accurately predict the direction of the Dow Jones Industrial Average (DJIA) by analyzing the news sentiment? 

# Hypothesis
News headlines drive the near-term performance of the stock market. By analyzing major news headlines around the world on any given day, we should be able to predict the direction of the stock market on that day. 

# How We Performed the Analysis
Data was provided by Kaggle. The dataset contained 27 total columns:

- Column 0: "Date"
- Column 1: "Label" reflecing the performance of the DJIA. Up days represented as '0' on a down day and '1' on an up day.
- Columns 2-26: "Top1, Top2...Top25" reflecing the top 25 headlines. The top 25 news headlines were sourced from the Reddit WorldNews Channel and voted on by Reddit users. 

# Initial DataFrame 

![alt text](Images/initital_df.png)

# Model Summary
We evaluated two different machine learning models: 

1. LSTM RNN using the the Keras Tokenizer class to vectorize the text

2. Random Forest using Scikit-learn's CountVectorizer class to represent text as bigrams (n=2) 

We chose these two models because they use different types of algorithms. 

Random Forest is an algorithm that uses bagging (selects a subset of data) to create a number of weak learners. It uses a subset of those weak learners and each of those features are chosen randomly. The model combines those weak learners to create a stronger classifier that is going to be less prone to overfitting. 

RNN (recurrent neural network) is able to factor in historical states and values and works with neural net architecture to come up with predicted values. LSTM RNN works like an original RNN, but it selects which types of longer-term events are worth remembering and which are okay to forget. 

# Data Cleanup & Exploration
The most difficult part of the project was finding a good dataset. Initially we wanted to pull in news articles from an API but most of the popular, free APIs (News Api, Stock News API, Yahoo Finance API powered by Rapid API etc.) had limitations with how much data you could pull or how many API calls you were allowed to perform. We also tried to learn how to scrape the web to pull data from SeekingAlpha but the site prohibits news scraping (HTTP error 403). 

# Data Cleanup

The headlines in the Kaggle dataset contained punctuation and had to be cleaned up in order to do the analysis. We converted all the headlines to regular expressions:

![alt text](Images/headline_regex.png)

We then converted all of the cells to lowercase strings so the text would be treated equally when implementing the bag of words for the Random Forest Model:

![alt text](Images/convert_lowercase.png)

We then joined all the news sources across all the rows to form one giant string:

![alt text](Images/join_news.png)

With all the news headlines joined togther in one column we then dropped all the individual news columns. The DataFrame is ready to train the models:

![alt text](Images/new_df.png)

# LSTM Models
We ran two LSTM models by varying the architecture for each. 

Model 1 had 280 units and 50 epochs: 

![alt text](Images/lstm_1.png)

Model 2 had 50 units and 30 epochs. We also added a 20% dropout rate:

![alt text](Images/lstm_2.png)

# LSTM Model 1 Evaluation

![alt text](Images/loss_function_1.png)
![alt text](Images/auc_1.png)
![alt text](Images/accuracy_1.png)

![alt text](Images/lstm_class_1.png)

# LSTM Model 2 Evaluation

![alt text](Images/loss_function_2.png)
![alt text](Images/auc_2.png)
![alt text](Images/accuracy_2.png)

![alt text](Images/lstm_class_2.png)

# LSTM RNN Model Performance
Both of the models performed poorly. The accuracy was only 0.49 in Model 1 and 0.50 in Model 2. 
Both models were clearly overfit; the models were performing too well on the training data and performing very poorly on the testing data. In other words, the models were picking up the noise or random fluctuations in the training data but these concepts did not apply to new data. In Model 2 the loss per iteration was increasing over time when in fact we were hoping for the exact opposite.

Despite varying the architecture considerably in both models they are clearly both not appropiate for any actual stock trading scenario.

# Random Forest Model Evaluation:

Applying a bag of words using CountVectorizer from the Scikit-learn library and then using a Random Forest classifier. 

More specifically, we used ngram_range = (2,2) and then applied fit_transform on the total news headlines. After that we applied the Random Forest classifier on the training dataset and training label. Finally we use the same transformation on the test dataset.  

![alt text](Images/rf_model.png)

# Random Forest Model Performance

![alt text](Images/rf_class.png)

The Random Forest model performed much better with an accuracy score of 86%. This means that if we get the news headlines for tomorrow and run this in our model, we should be able to accurately predict whether the DJIA is up or down 86% of the time. 

# Discussion
It was interesting to see that the two models performed so differently. We assumed LSTM model would perform better because we could adjust the number of hidden layers and add a dropout rate. Not only were both the LSTM models overfit but in one of the LSTM models, the loss per iteration *increased* over time. 
The Random Forest using a bag of words performed markedly better than the LSTM model using sentiment analysis. The Random Forest model had an accuracy score of 0.86, indicating that we were able to predict the direction of the DJIA 86% of the time. 

# Post Mortem
The most difficult part of the project was finding a good dataset. Initially we wanted to pull in news articles from an API but most of the popular, free APIs (News Api, Stock News API, Yahoo Finance API powered by Rapid API etc.) had limitations with how much data you could pull or how many API calls you were allowed to make. We also tried to scrape the web to pull data from SeekingAlpha but the site prohibits news scraping (HTTP error 403). 
Given tokenization and vectorization only categorize language, had we have more time we would have implemented a sentiment analysis such as vader to provide additional context to our model for machine learning. 
If we had been able to pull in real time news data from an API, we would have liked to feed it into one of our models to run an algo trading platform and see how it actually performed. It would also be interesting to see if we could build a model that predicted the actual return of the market versus just the directionality of the market.
