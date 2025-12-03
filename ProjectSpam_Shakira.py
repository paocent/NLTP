import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from collections import Counter

# --- ⚠️ IMPORTANT: Run these lines once if you get an NLTK error ⚠️ ---
# import warnings
# warnings.filterwarnings("ignore")
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# --- 🛠️ NLP Preprocessing Function Setup with Fallback ---
try:
    # Attempt to use the full NLTK setup (preferred)
    lemmatizer = WordNetLemmatizer()
    english_stopwords = set(stopwords.words('english'))


    def custom_tokenizer_clean(text):
        """ Cleans, tokenizes, removes stopwords, and lemmatizes text (Full NLTK version). """
        # 1. Remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        # 2. Tokenize (split into words)
        tokens = nltk.word_tokenize(text)
        # 3. Lemmatize, removing stopwords and short tokens
        lemmas = [
            lemmatizer.lemmatize(w)
            for w in tokens
            if w not in english_stopwords and len(w) > 2
        ]
        return lemmas


    def log_stopwords_in_data(data_series, stopwords_set):
        """ Logs which stop words are present in the dataset. """
        removed_words_counter = Counter()
        for comment in data_series:
            text = re.sub(r'[^a-zA-Z\s]', '', comment.lower())
            tokens = nltk.word_tokenize(text)
            for word in tokens:
                if word in stopwords_set:
                    removed_words_counter[word] += 1
        return removed_words_counter

except Exception:
    # Fallback to simple tokenizer if NLTK resources are missing
    print("\n--- ⚠️ NLTK WARNING: Falling back to a simple tokenizer (no stop word removal/lemmatization). ---\n")
    english_stopwords = set()  # Empty set for display purposes


    def custom_tokenizer_clean(text):
        """ Simplified tokenizer: lowercases, removes non-alphabetic chars, and splits into words. """
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        return [w for w in text.split() if len(w) > 2]


    def log_stopwords_in_data(data_series, stopwords_set):
        return Counter({'NLTK': 0})  # Dummy counter for compatibility


# --------------------------------------------------
# --- NEW: Feature Engineering for Spam Indicators (Links & HTML) ---

def preprocess_for_spam_features(text):
    """
    Detects and tags links and HTML content before standard tokenization.
    """

    # 1. Detect and tag URLs/Links (http, https, www)
    # Replaces the URL with a special tag
    link_pattern = r'(http|www)\S+'
    if re.search(link_pattern, text, re.IGNORECASE):
        text = re.sub(link_pattern, '__HAS_LINK__', text)

    # 2. Detect and tag HTML tags (<tag>...</tag> or just <a>)
    # Replaces the HTML tag with a special tag
    html_pattern = r'<[^>]+>'
    if re.search(html_pattern, text):
        text = re.sub(html_pattern, '__HAS_HTML__', text)

    # Return the modified text for the standard tokenizer to handle the rest
    return text


def custom_tokenizer(text):
    """
    Combined tokenizer: applies spam feature detection first, then cleans and tokenizes.
    """
    # 1. Apply spam feature detection (links/html tags)
    text = preprocess_for_spam_features(text)

    # 2. Apply standard cleaning and tokenization
    tokens = custom_tokenizer_clean(text)

    # 3. Add the special tags as tokens if they were found and replaced
    # We must ensure the special tags are included in the token list for CountVectorizer
    if '__HAS_LINK__' in text:
        tokens.append('__HAS_LINK__')
    if '__HAS_HTML__' in text:
        tokens.append('__HAS_HTML__')

    return tokens


# --- NEW: Feature Importance Function (Explanation) ---

def get_top_spam_features(model, vectorizer, input_tfidf, top_n=5):
    """
    Identifies the top N words in the input that contribute most to the SPAM prediction.
    """
    spam_log_probs = model.feature_log_prob_[1]
    input_indices = input_tfidf.nonzero()[1]

    # Check if there are any features present in the input
    if len(input_indices) == 0:
        return pd.Series()

    input_tfidf_scores = input_tfidf[0, input_indices].toarray().flatten()
    feature_scores = input_tfidf_scores * spam_log_probs[input_indices]
    feature_names = np.array(vectorizer.get_feature_names_out())[input_indices]
    scored_features = pd.Series(feature_scores, index=feature_names)

    # Sort and return the words that have the highest influence on the SPAM prediction
    return scored_features.nlargest(top_n)


# --------------------------------------------------

## 📚 Data Loading and Initial Exploration

# --- 1. Load the data into a pandas data frame ---
print("## 📚 Data Loading")
# NOTE: Ensure 'Youtube05-Shakira.csv' is in the same directory as this script!
try:
    datasets = pd.read_csv("Youtube05-Shakira.csv", encoding='latin-1')
    datasets = datasets[['CONTENT', 'CLASS']].rename(columns={'CONTENT': 'content', 'CLASS': 'class'})
    print(f"Dataset loaded successfully with {len(datasets)} rows.")
except FileNotFoundError:
    print("FATAL ERROR: 'Youtube05-Shakira.csv' not found. Please ensure the file is in the script directory.")
    exit()
print("-" * 30)

## ☁️ Initial Word Cloud Generation (Unfiltered)
print("## ☁️ Initial Word Cloud (Raw Data)")
text = " ".join(datasets['content'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Frequent Words in the RAW Dataset")
plt.show()
print("Initial Word Cloud displayed.")
print("-" * 30)

# --- Focused Word Cloud ---

## ☁️ Focused Word Cloud (Ham-Only, Stop Words Removed)
print("## ☁️ Focused Word Cloud (Ham-Only)")
datasets_ham = datasets[datasets['class'] == 0]
all_ham_words = []
for comment in datasets_ham['content']:
    # Use the cleaning part of the tokenizer for the word cloud
    all_ham_words.extend(custom_tokenizer_clean(comment))
cleaned_ham_text = " ".join(all_ham_words)
focused_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_ham_text)
plt.figure(figsize=(10, 5))
plt.imshow(focused_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Frequent Words in CLEANED HAM Comments")
plt.show()
print("Focused Word Cloud (Ham-only) displayed.")
print("-" * 30)

# --------------------------------------------------

## 🛑 Stop Word Analysis 🛑

print("## 🛑 Stop Word Analysis")
if len(english_stopwords) > 0:
    print(f"Total number of stop words in NLTK's English list: **{len(english_stopwords)}**")
    removed_counts = log_stopwords_in_data(datasets['content'], english_stopwords)
    print("\n**Top 10 Most Frequently Removed Stop Words in this Dataset:**")
    top_removed = removed_counts.most_common(10)
    top_removed_df = pd.DataFrame(top_removed, columns=['Stop Word', 'Count']).to_markdown(index=False)
    print(top_removed_df)
else:
    print("Skipping Stop Word Analysis due to NLTK resource failure.")
print("-" * 30)

# --------------------------------------------------

## ⚙️ Feature Engineering: Vectorization and TF-IDF

# --- 2 & 3. Feature Engineering: CountVectorizer (Bag-of-Words) with Lemmatization ---
print("## ⚙️ Feature Engineering (Bag-of-Words & Tokenization)")
# NOTE: Using the new combined 'custom_tokenizer' here
count_vectorizer = CountVectorizer(tokenizer=custom_tokenizer)
X_counts = count_vectorizer.fit_transform(datasets['content'])
print(f"New shape of the data after Bag-of-Words/Tokenization: **{X_counts.shape}**")
print(f"Total number of unique features/words: **{X_counts.shape[1]}**")
print("-" * 30)

# --- 4 & 5. Downscaling: TF-IDF Transformer (Final Features) ---
print("## 📉 Downscaling (TF-IDF Transformer)")
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
print(f"New shape of the data after TF-IDF: **{X_tfidf.shape}**")
print("-" * 30)

# --------------------------------------------------

## 📝 Exploratory Analysis: Frequent Word Insights

print("## 📝 Frequent Word Analysis (Top 10)")
datasets_spam = datasets[datasets['class'] == 1]


def get_top_words(data_subset, vectorizer, top_n=10):
    """Calculates the top N most frequent words in a text data subset."""
    counts = vectorizer.transform(data_subset['content'])
    word_freq = np.sum(counts.toarray(), axis=0)
    feature_names = np.array(vectorizer.get_feature_names_out())
    freq_series = pd.Series(word_freq, index=feature_names)
    top_words = freq_series.nlargest(top_n)
    return top_words


top_ham_words = get_top_words(datasets_ham, count_vectorizer)
print("\n**Top 10 Ham (Legitimate) Words:**")
print(top_ham_words.to_markdown(numalign="left", stralign="left"))

top_spam_words = get_top_words(datasets_spam, count_vectorizer)
print("\n**Top 10 Spam Words:**")
print(top_spam_words.to_markdown(numalign="left", stralign="left"))
print("-" * 30)

# --------------------------------------------------

## 🔀 Data Splitting and Model Training

# --- 6 & 7. Shuffling and Splitting Data ---
print("## 🔀 Shuffling and Splitting Data")
X_df = pd.DataFrame(X_tfidf.toarray())
y_df = datasets['class'].reset_index(drop=True)
combined_df = pd.concat([X_df, y_df], axis=1)

shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
split_point = int(0.75 * len(shuffled_df))

X = shuffled_df.drop(columns=['class'])
y = shuffled_df['class']

X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]
y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

print(f"Training set size: {len(X_train)} samples (75%)")
print(f"Testing set size: {len(X_test)} samples (25%)")
print("-" * 30)

# --- 8. Fit the training data into a Naive Bayes classifier ---
print("## 🧠 Model Training (Multinomial Naive Bayes)")
model = MultinomialNB()
model.fit(X_train, y_train)
print("Multinomial Naive Bayes model trained successfully.")

# --------------------------------------------------

## 🎯 Model Evaluation

# --- 9. Cross validate the model on the training data using 5-fold ---
print("\n## 🎯 Model Evaluation")
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"5-Fold Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean 5-Fold Cross-Validation Accuracy: **{cv_scores.mean():.4f}**")

# --- 10. Test the model on the test data and calculate metrics ---
y_pred = model.predict(X_test)

metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1-Score': f1_score(y_test, y_pred, zero_division=0)
}

print("\n**Performance Metrics (Test Data):")
for name, value in metrics.items():
    print(f"- {name}: **{value:.4f}**")

conf_matrix = confusion_matrix(y_test, y_pred)
print("\n**Confusion Matrix (Test Data):**")
print(conf_matrix)

# --------------------------------------------------

## 📈 Performance Visualization (The Two Requested Charts)

print("\n## 📈 Performance Visualization")

# --- 1. Confusion Matrix Heatmap ---
plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = ['HAM (0)', 'SPAM (1)']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# --- 2. Metrics Bar Chart ---
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

plt.figure(figsize=(8, 5))
bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
plt.ylim(0, 1.0)
plt.title('Multinomial Naive Bayes Model Performance (Test Set)')
plt.ylabel('Score')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
print("Visualization complete.")
print("-" * 30)

# --------------------------------------------------

## 💬 Interactive Classification Loop

# --- 12. User Input Classification Loop ---
print("## 💬 User Input Classification Loop")

while True:
    user_input_comment = input("\nEnter a YouTube comment to classify (type 'exit' to quit):\n> ")

    if user_input_comment.lower() == 'exit':
        print("\nExiting comment classification. Goodbye!")
        break

    # 1. Vectorize the new text (uses the new custom_tokenizer)
    new_counts = count_vectorizer.transform([user_input_comment])

    # 2. Transform the counts to TF-IDF weights
    new_tfidf = tfidf_transformer.transform(new_counts)

    # 3. Predict the class
    new_prediction = model.predict(new_tfidf.toarray())[0]

    # Print the result
    result_label = "**SPAM (Class 1)** 🚨" if new_prediction == 1 else "**HAM (Class 0)** ✅"
    print("\n--- CLASSIFICATION RESULT ---")
    print(f"Input Comment: '{user_input_comment}'")
    print(f"Predicted Class: {result_label}")

    # 4. Feature Explanation
    if new_prediction == 1:
        # Only analyze if predicted as spam
        top_features = get_top_spam_features(model, count_vectorizer, new_tfidf)
        print("\n### 🚨 Reason for SPAM Detection (Top 5 Keywords):")
        print(
            "This classification is based on the following words/features having the highest weights for the SPAM class:")

        if top_features.empty:
            print("- The input comment contains no recognized features in the model's vocabulary.")
        else:
            for word, score in top_features.items():
                # Highlight link/html feature
                if word in ['__HAS_LINK__', '__HAS_HTML__']:
                    display_word = f"**{word.replace('_', '')}**"
                else:
                    display_word = word.upper()

                print(f"- {display_word} (Score: {score:.4f})")

    print("-----------------------------")