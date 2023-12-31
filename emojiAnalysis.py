

def analyze_emoji_sentiment(emoji_df):
    # Define a mapping of emojis to sentiment categories
    sentiment_mapping = {
        "😄": "positive",
        "😃": "positive",
        "😀": "positive",
        "😁": "positive",
        "😆": "positive",
        "😅": "positive",
        "😂": "positive",
        "🙂": "positive",
        "🙃": "positive",

        "😔": "negative",
        "🙁": "negative",
        "☹️": "negative",
        "😞": "negative",
        "😟": "negative",
        "😣": "negative",
        "😖": "negative",
        "😢": "negative",
        "😭": "negative",

        "😐": "neutral",
        "😶": "neutral",
        "😑": "neutral",
        "😬": "neutral",
        "😮": "neutral",
        "😯": "neutral",
        "😕": "neutral",
        "😷": "neutral",
        "🤐": "neutral",

        "🥲": "neutral",
        "😓": "neutral",
        "😱": "neutral",
        "🥳": "positive",
        "❗": "positive",
        "🥰": "positive",
        "🤓": "neutral",
        "🤔": "neutral",
        "👀": "neutral",
        "🤗": "positive",
        "❤": "positive",
        "👏": "positive",
        "🍰": "positive",
        "💯": "positive",
        "🍦": "positive",
        "🤩": "positive",
        "🫡": "neutral",

        "❗": "positive",
        "▪": "neutral",
        "‼": "positive",
        "📌": "neutral",
        "🌟": "positive",
        "⭐": "positive",
        "🗓": "neutral",
        "📚": "neutral",
        "🎯": "positive",
        "🫶": "neutral",
        "🏼": "neutral",
        "🥳": "positive",
        "💰": "positive",
        "🏆": "positive",
        "📢": "positive",
        "🔗": "neutral",
        "🎉": "positive",
        "🤝": "positive",
        "🎓": "positive",
        "👍": "positive",
        "📣": "positive",
        "🏢": "neutral",
        "🔬": "neutral",
        "📖": "neutral",
        "🥈": "positive",
        "🥉": "positive",
        "🎫": "positive",
        "✅": "positive",
        "📃": "neutral",
        "👋": "positive",
        "☝": "neutral",
        "🏻": "neutral",
        "✋": "neutral",
        "📝": "neutral",
        "🔑": "neutral",
        "🚀": "positive",
        "🚨": "neutral",

        # Add more emojis and their sentiment categories as needed
        "😅": "positive",
        "😂": "positive",
        "🤣": "positive",
        "😊": "positive",
        "😇": "positive",
        "😁": "positive",
        "😀": "positive",
        "🥰": "positive",
        "😍": "positive",
        "🤩": "positive",
        "😘": "positive",
        "😗": "positive",
        "😚": "positive",
        "😌": "positive",
        "😋": "positive",
        "😛": "positive",
        "😜": "positive",
        "😝": "positive",
        "🤑": "positive",
        "🤗": "positive",
        "🙌": "positive",
        "👏": "positive",
        "👍": "positive",
        "👌": "positive",
        "👊": "positive",
        "✊": "positive",
        "🤛": "positive",
        "🤜": "positive",
        "🤞": "positive",
        "✌": "positive",
        "🤘": "positive",
        "🤙": "positive",
        "👋": "positive",
        "🤚": "positive",
        "🖐": "positive",
        "✋": "positive",
        "👆": "positive",
        "👇": "positive",
        "👈": "positive",
        "👉": "positive",
        "🖕": "positive",
        "🤘": "positive",
        "🖖": "positive",
        "🙏": "positive",
        "🤞": "positive",
        "🤟": "positive",
        "🤝": "positive",
        "🧡": "positive",
        "💛": "positive",
        "💚": "positive",
        "💙": "positive",
        "💜": "positive",
        "💖": "positive",
        "💗": "positive",
        "💘": "positive",
        "💝": "positive",
        "💞": "positive",
        "💟": "positive",
        "❣": "positive",
        "💌": "positive",
        "💤": "positive",

        # Add more emojis and their sentiment categories as needed
    }

    # Initialize counters for sentiment categories
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    # Iterate through emojis in the DataFrame and categorize them
    for emojis, count in emoji_df.values:
        sentiment = sentiment_mapping.get(emojis, "unknown")
        if sentiment == "positive":
            positive_count += count
        elif sentiment == "negative":
            negative_count += count
        elif sentiment == "neutral":
            neutral_count += count

    return negative_count, neutral_count, positive_count
