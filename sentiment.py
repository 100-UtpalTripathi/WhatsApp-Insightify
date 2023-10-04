import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data (if not already downloaded)
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

def find_sentiment(chat_data):
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        # Tokenization
        words = word_tokenize(text.lower())

        # Remove punctuation and stopwords
        words = [word for word in words if word.isalnum() and word not in stop_words]

        return ' '.join(words)

    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()

    sentiments = []
    for message in chat_data:
        preprocessed_message = preprocess_text(message)
        sentiment_score = analyzer.polarity_scores(preprocessed_message)

        # Classify sentiment based on compound score
        if sentiment_score['compound'] >= 0.05:
            sentiment = "Positive"
        elif sentiment_score['compound'] <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        sentiments.append(sentiment)
    return sentiments
    # Count Sentiments
    #sentiment_counts = Counter(sentiments)

    # Visualization
    # plt.bar(sentiment_counts.keys(), sentiment_counts.values())
    # plt.xlabel('Sentiment')
    # plt.ylabel('Count')
    # plt.title('Sentiment Analysis of Chat Messages')
    # plt.show()