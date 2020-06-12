Tales from the Crypto - Natural Language Processing

![Stock Sentiment](Images/sentimental.jpeg)

## Background

As there's been a lot of hype in the news lately about cryptocurrency, we would like to invest, so to speak, of the latest news headlines regarding Bitcoin & Ethereum to get a better feel for the current public sentiment around each coin.

Using fundamental NLP techniques to understand the sentiment in the latest news article featuring Bitcoin & Ethereum and also other factors involved with the coin prices such as common words & phrases and organizations & entities mentioned in the articles.

Following tasks are performed:

1. [Sentiment Analysis](#Sentiment-Analysis)
2. [Natural Language Processing](#Natural-Language-Processing)
3. [Named Entity Recognition](#Named-Entity-Recognition)

- - -

### Files

[Starter Notebook](Starter_Code/crypto_sentiment.ipynb)

- - -
<details>

<summary> Sentiment Analysis </summary>

Use of [newsapi](https://newsapi.org/) to pull the latest news articles for Bitcoin and Ethereum and created DataFrame of sentiment scores for each coin.

Descriptive statistics:
    <ul>
    <li> Which coin had the highest mean positive score? </li>
        ```
        Ethereum - 0.08 
        ```
    </ul>
    <ul>
    <li> Which coin had the highest negative score? </li>
        ```
        Ethereum - 0.025 
        ```
    </ul>
    <ul>
    <li> Which coin had the highest positive score? </li>
        ```
        Ethereum - 0.9198 
        ```
    </ul>
    
</details>
    
<details>

<summary> Natural Language Processing </summary>
* Import the following Libraries from nltk:
    ```
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from string import punctuation
    import re
    ```

  
* Use NLTK and Python to tokenize the text for each coin
<br>1. Remove punctuation</br>
    ```
    regex = re.compile("[^a-zA-Z0-9 ]")
re_clean = regex.sub('', text)
    ```
<br>2. Lowercase each word</br>
    ```
    words = word_tokenize(re_clean.lower())
    ```
<br>3. Remove stop words</br>
     ```
     sw = set(stopwords.words('english'))
     ```
<br>4. Lemmatize Words into Root words</br>
    ```
  lemmatizer = WordNetLemmatizer()
lem = [lemmatizer.lemmatize(word) for word in words]
    ```

* Next, look at the ngrams and word frequency for each coin.

1. Use NLTK to produce the ngrams for N = 2.
2. List the top 10 words for each coin.

Finally, generate word clouds for each coin to summarize the news for each coin.

![btc-word-cloud.png](Images/btc-word-cloud.png)

![eth-word-cloud.png](Images/eth-word-cloud.png)



#### Named Entity Recognition

In this section, you will build a named entity recognition model for both coins and visualize the tags using SpaCy.

![btc-ner.png](Images/btc-ner.png)

![eth-ner.png](Images/eth-ner.png)

- - -

### Resources

[Vader Sentiment Analysis](http://www.nltk.org/howto/sentiment.html)

- - -

