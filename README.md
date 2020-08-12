# Columbia Fintech Bootcamp - Project 2
*Assignment - Use Jupyter, Google Colab, or AWS SageMaker to prepare a training and testing dataset and to train the machine-learning model.*
*Create one or more machine learning models.

# Performing Sentiment Analysis to Predict the Direction of the Dow Jones Industrial Average

## Team Members:
- David Still
- Charles Xia
- Ian Walter

# Topic of the Analysis:
Can we accurately predict the direction of the Dow Jones Industrial Average (DJIA) by analyzing the news sentiment? 

# Hypothesis
News headlines drive the near-term performance of the stock market. By analyzing major news headlines around the world on any given day, we should be able to predict the direction of the stock market on that day. 

# How We Performed the Analysis
Data was provided by Kaggle. The dataset contains 27 total columns:

- Column 1: Date
- Column 2: Performance of the DJIA. Up days represented as '0' on a down day and '1' on an up day.
- Columns 3-27: Top 25 news headlines. The top 25 news headlines were sourced from the Reddit WorldNews Channel and voted on by Reddit users. 

# Model Summary
Two different machine learning models 
- Random Forest using bag-of-words to represent text as bigrams (n=2) represents the frequency of the tokens and we used n-grams for the model.
- LSTM RNN using tokenization to split the string into tokens Tokenization splits the string into tokens and gives us a sequence of tokens. 


# Data Cleanup & Exploration
The most difficult part of the project was finding a good dataset. Initially we wanted to pull in news articles from an API but most of the popular, free APIs (News Api, Stock News API, Yahoo Finance API powered by Rapid API etc.) had limitations with how much data you could pull or how many API calls you were allowed to perform. We also tried to learn how to scrape the web to pull data from SeekingAlpha but the site prohibits news scraping (HTTP error 403). 

The Kaggle dataset fortunately was 


URL for Google CoLab: https://colab.research.google.com/drive/1Z8Sg5yBEaz8Z3iidxGxoqjKTyG0HBPmv?usp=sharing
URL for Presentation: https://docs.google.com/presentation/d/1OZdSwFY6oAaGBmVpiXtSWOTG1iQGQKcTEvzLdvL_t8I/edit?usp=sharing

## Project Proposal
This project is to utilize news sentiment to predict stock movement. The use of Natural Language Processing (NLP) will generate positive sentiment of news gathered from Reddit's Top 25 headlines. Machine learning models will be run to see if we can predict future good or bad days of the Dow Jones Industrial Average (DJIA). We will be sourcing our data from the a Kaggle dataset and Yahoo Finance.

## Data/Apis Used
- Kaggle DJIA Dataset

## Machine Learning Models Used
- Random Forest
- LSTM RNN

## Technologies Used
- Google Colab
- Pandas
- Numpy
- Matplotlib

