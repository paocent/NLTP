# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 20:00:30 2025

@author: three
"""

# youtube_spam_dashboard.py
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# wordcloud
from wordcloud import WordCloud

# nltk (optional; fallback implemented)
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --------------------------
# NLTK setup with fallback
# --------------------------
try:
    # Attempt NLTK downloads silently if not present (commented out to avoid repeated downloads; enable if needed)
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    english_stopwords = set(stopwords.words('english'))

    def custom_tokenizer_clean(text):
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        tokens = nltk.word_tokenize(text)
        lemmas = [lemmatizer.lemmatize(w) for w in tokens if w not in english_stopwords and len(w) > 2]
        return lemmas

    def log_stopwords_in_data(data_series, stopwords_set):
        removed_words_counter = Counter()
        for comment in data_series.astype(str):
            text = re.sub(r'[^a-zA-Z\s]', '', comment.lower())
            tokens = nltk.word_tokenize(text)
            for word in tokens:
                if word in stopwords_set:
                    removed_words_counter[word] += 1
        return removed_words_counter

except Exception:
    print("\n--- ⚠️ NLTK resources unavailable — using fallback tokenizer (no stopword removal/lemmatization). ---\n")
    english_stopwords = set()

    def custom_tokenizer_clean(text):
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        return [w for w in text.split() if len(w) > 2]

    def log_stopwords_in_data(data_series, stopwords_set):
        return Counter()

# --------------------------------------------------
# Spam-specific preprocessing and combined tokenizer
# --------------------------------------------------
def preprocess_for_spam_features(text):
    text = str(text)
    # Tag links
    link_pattern = r'(http[s]?://\S+|www\.\S+)'
    if re.search(link_pattern, text, re.IGNORECASE):
        text = re.sub(link_pattern, ' __HAS_LINK__ ', text)
    # Tag HTML
    html_pattern = r'<[^>]+>'
    if re.search(html_pattern, text):
        text = re.sub(html_pattern, ' __HAS_HTML__ ', text)
    return text

def custom_tokenizer(text):
    text = preprocess_for_spam_features(text)
    tokens = custom_tokenizer_clean(text)
    # Ensure tags included
    if '__HAS_LINK__' in text:
        tokens.append('__HAS_LINK__')
    if '__HAS_HTML__' in text:
        tokens.append('__HAS_HTML__')
    return tokens

# --------------------------------------------------
# Feature importance helper (top spam features for a single input)
# --------------------------------------------------
def get_top_spam_features(model, vectorizer, input_tfidf, top_n=5):
    # model.feature_log_prob_[1] gives log prob for spam class
    try:
        spam_log_probs = model.feature_log_prob_[1]
    except Exception:
        return pd.Series()
    input_indices = input_tfidf.nonzero()[1]
    if len(input_indices) == 0:
        return pd.Series()
    input_tfidf_scores = input_tfidf[0, input_indices].toarray().flatten()
    feature_scores = input_tfidf_scores * spam_log_probs[input_indices]
    feature_names = np.array(vectorizer.get_feature_names_out())[input_indices]
    scored_features = pd.Series(feature_scores, index=feature_names)
    return scored_features.nlargest(top_n)

# --------------------------
# Load dataset
# --------------------------
DATAFILE = "Youtube05-Shakira.csv"
if not os.path.exists(DATAFILE):
    raise FileNotFoundError(f"Dataset file '{DATAFILE}' not found. Put it in the same directory as this script.")

df = pd.read_csv(DATAFILE, encoding='latin-1')
df = df[['CONTENT', 'CLASS']].rename(columns={'CONTENT': 'content', 'CLASS': 'class'})
print(f"Loaded dataset with {len(df)} rows.")

# --------------------------
# Initial wordcloud (raw)
# --------------------------
raw_text = " ".join(df['content'].astype(str))
wc_raw = WordCloud(width=800, height=400, background_color='white').generate(raw_text)
plt.figure(figsize=(10, 4))
plt.imshow(wc_raw, interpolation='bilinear')
plt.axis('off')
plt.title("Raw Dataset - Most Frequent Words")
plt.show()

# Focused wordcloud for ham (cleaned)
ham_df = df[df['class'] == 0]
all_ham_tokens = []
for c in ham_df['content']:
    all_ham_tokens.extend(custom_tokenizer_clean(c))
wc_ham = WordCloud(width=800, height=400, background_color='white').generate(" ".join(all_ham_tokens))
plt.figure(figsize=(10, 4))
plt.imshow(wc_ham, interpolation='bilinear')
plt.axis('off')
plt.title("Ham Comments - Cleaned (Stop words removed if NLTK available)")
plt.show()

# --------------------------
# Stop word analysis
# --------------------------
print("\n--- Stop Word Analysis ---")
if english_stopwords:
    sw_counts = log_stopwords_in_data(df['content'], english_stopwords)
    top_removed = sw_counts.most_common(10)
    if top_removed:
        print("Top 10 stop words found in dataset (counts):")
        print(pd.DataFrame(top_removed, columns=['stop_word', 'count']).to_string(index=False))
    else:
        print("No stopwords were counted (unexpected).")
else:
    print("NLTK stopwords not available; skipping detailed stopword analysis.")

# --------------------------
# Vectorization: CountVectorizer + TF-IDF
# --------------------------
cv = CountVectorizer(tokenizer=custom_tokenizer)
X_counts = cv.fit_transform(df['content'])
tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_counts)

print(f"Feature matrix shape: {X_tfidf.shape}")

# --------------------------
# Top words (top 10 ham/spam)
# --------------------------
def get_top_words_by_subset(subset_df, vectorizer, top_n=10):
    counts = vectorizer.transform(subset_df['content'])
    word_freq = np.sum(counts.toarray(), axis=0)
    feature_names = np.array(vectorizer.get_feature_names_out())
    s = pd.Series(word_freq, index=feature_names).nlargest(top_n)
    return s

top_ham = get_top_words_by_subset(ham_df, cv, 10)
top_spam = get_top_words_by_subset(df[df['class'] == 1], cv, 10)

print("\nTop 10 Ham Words:\n", top_ham.to_string())
print("\nTop 10 Spam Words:\n", top_spam.to_string())

# --------------------------
# Shuffle & split (75/25)
# --------------------------
X_df = pd.DataFrame(X_tfidf.toarray())
y = df['class'].reset_index(drop=True)
combined = pd.concat([X_df, y], axis=1).sample(frac=1, random_state=42).reset_index(drop=True)

split_idx = int(0.75 * len(combined))
X = combined.drop(columns=['class']).values
y = combined['class'].values
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# --------------------------
# Train model
# --------------------------
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model trained.")

# Cross validation on training set (5-fold)
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print("5-fold CV scores:", cv_scores)
cv_mean = cv_scores.mean()

# Test predictions & metrics
y_pred = model.predict(X_test)
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1-Score': f1_score(y_test, y_pred, zero_division=0)
}
print("\nTest set metrics:")
for k, v in metrics.items():
    print(f" - {k}: {v:.4f}")

conf_mat = confusion_matrix(y_test, y_pred)

# --------------------------
# Build dashboard with matplotlib (layout similar to screenshot)
# --------------------------
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[1,1,1,0.7], height_ratios=[1,1,1])
fig.suptitle("YouTube Spam Comment Classifier - Complete Analysis", fontsize=18, fontweight='bold')

# Confusion Matrix (top-left)
ax0 = fig.add_subplot(gs[0, 0])
im = ax0.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Reds)
ax0.set_title("Confusion Matrix")
ax0.set_xticks([0,1]); ax0.set_xticklabels(['HAM (0)','SPAM (1)'], rotation=45)
ax0.set_yticks([0,1]); ax0.set_yticklabels(['HAM (0)','SPAM (1)'])
thresh = conf_mat.max() / 2.
for i in range(conf_mat.shape[0]):
    for j in range(conf_mat.shape[1]):
        ax0.text(j, i, format(conf_mat[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_mat[i, j] > thresh else "black")
fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

# Model Performance Comparison (CV mean vs Test accuracy)
ax1 = fig.add_subplot(gs[0, 1])
bars = ax1.bar(['CV Mean', 'Test Accuracy'], [cv_mean, metrics['Accuracy']], alpha=0.9)
ax1.set_ylim(0.8, 1.0)
ax1.set_title("Model Performance Comparison")
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005, f"{bar.get_height():.3f}", ha='center')

# Original dataset distribution (pie)
ax2 = fig.add_subplot(gs[0, 2])
counts = df['class'].value_counts().sort_index()
labels = ['Not Spam', 'Spam']
ax2.pie(counts, labels=labels, autopct='%1.0f%%', startangle=140)
ax2.set_title("Original Dataset Distribution")

# Cross-validation performance (line)
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(range(1, len(cv_scores)+1), cv_scores, marker='o', linestyle='-', linewidth=2)
ax3.axhline(cv_mean, color='red', linestyle='--', label=f"Mean: {cv_mean:.3f}")
ax3.set_xlabel("Fold Number")
ax3.set_ylabel("Accuracy")
ax3.set_title("Cross-Validation Performance (5-Fold)")
ax3.set_ylim(0.0, 1.05)
ax3.legend()

# Custom comments predictions (bar) - initial placeholder (will update during interactive loop)
custom_counts = {'Correct': 0, 'Incorrect': 0, 'Skipped': 0}
ax4 = fig.add_subplot(gs[1, 1])
bars_custom = ax4.bar(['Correct'], [custom_counts['Correct']], color='lightgreen')
ax4.set_ylim(0, 10)
ax4.set_title("Custom Comments Predictions (Correct)")

# Top words plots (ham/spam)
ax5 = fig.add_subplot(gs[1, 2])
top_spam.plot(kind='bar', ax=ax5)
ax5.set_title("Top 10 Spam Words (Counts from CountVectorizer)")
ax5.set_ylabel("Raw Count")

ax6 = fig.add_subplot(gs[2, 0])
top_ham.plot(kind='bar', ax=ax6, color='green')
ax6.set_title("Top 10 Ham Words (Counts from CountVectorizer)")

# Class distribution (bar + pie)
ax7 = fig.add_subplot(gs[2, 1])
class_counts = df['class'].value_counts().sort_index()
ax7.bar(['Not Spam','Spam'], class_counts.values, color=['green','red'])
ax7.set_title("Class Distribution (Counts)")
ax7.set_ylabel("Count")

ax8 = fig.add_subplot(gs[2, 2])
ax8.pie(class_counts, labels=['Not Spam','Spam'], autopct='%1.0f%%', startangle=90)
ax8.set_title("Class Distribution (%)")

# Model performance summary box (right column spanning rows)
summary_ax = fig.add_subplot(gs[:, 3])
summary_ax.axis('off')
summary_text = (
    "MODEL PERFORMANCE SUMMARY\n"
    "-------------------------\n"
    f"Dataset: {DATAFILE}\n"
    f"Total Samples: {len(df)}\n"
    f"Training Samples: {len(X_train)}\n"
    f"Testing Samples: {len(X_test)}\n\n"
    "CROSS-VALIDATION (5-Fold):\n"
    f"- Mean Accuracy: {cv_mean:.4f}\n"
    f"- Std Dev: {cv_scores.std():.4f}\n\n"
    "TEST SET PERFORMANCE:\n"
    f"- Accuracy: {metrics['Accuracy']:.4f}\n"
    f"- Precision: {metrics['Precision']:.4f}\n"
    f"- Recall: {metrics['Recall']:.4f}\n"
    f"- F1-Score: {metrics['F1-Score']:.4f}\n\n"
    "CONCLUSION:\nThe Multinomial Naive Bayes classifier performs well on this dataset.\n"
)
summary_ax.text(0, 1, summary_text, fontsize=10, va='top', family='monospace', bbox=dict(facecolor='#f7f1e0', edgecolor='black'))

plt.tight_layout(rect=[0, 0, 0.95, 0.96])
plt.show()

# Also produce a simple metrics bar chart (separate figure)
plt.figure(figsize=(7,4))
names = list(metrics.keys())
vals = [metrics[n] for n in names]
bars2 = plt.bar(names, vals)
plt.ylim(0,1)
plt.title("Test Metrics (bar)")
for b in bars2:
    plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{b.get_height():.3f}", ha='center')
plt.show()

# --------------------------
# Save top words tables for reference
# --------------------------
top_ham.to_csv("top_ham_words.csv")
top_spam.to_csv("top_spam_words.csv")
print("Saved top words CSVs: top_ham_words.csv, top_spam_words.csv")

# --------------------------
# Interactive classification loop (tracks results and saves to CSV)
# --------------------------
USER_RESULTS_CSV = "user_classification_results.csv"
# create CSV header if not exists
if not os.path.exists(USER_RESULTS_CSV):
    pd.DataFrame(columns=["comment", "predicted_class", "user_confirmed_correct", "true_label_if_provided"]).to_csv(USER_RESULTS_CSV, index=False)

print("\n=== Interactive Comment Classification ===")
print("Type 'exit' to quit. After the prediction, you'll be asked whether the prediction was correct; answer y/n/skip.")
print("If you know the true label, you can type it when asked (0 for ham, 1 for spam), or press Enter to leave blank.")

# maintain counts
correct_count = 0
incorrect_count = 0
skipped_count = 0

while True:
    user_comment = input("\nEnter a YouTube comment to classify (or 'exit'): \n> ")
    if user_comment.lower().strip() == 'exit':
        print("Exiting interactive loop.")
        break

    # vectorize using the same count_vectorizer and tfidf transformer
    new_counts = cv.transform([user_comment])
    new_tfidf = tfidf.transform(new_counts)
    pred = model.predict(new_tfidf)[0]
    label_text = "SPAM (1) 🚨" if pred == 1 else "HAM (0) ✅"
    print(f"\nPredicted class: {label_text}")

    # If spam, show top contributing features
    if pred == 1:
        top_feats = get_top_spam_features(model, cv, new_tfidf, top_n=6)
        if top_feats.empty:
            print("- No recognized features from vocabulary contributed to the spam prediction.")
        else:
            print("Top features contributing to SPAM prediction:")
            for w, s in top_feats.items():
                disp = w.upper() if not w.startswith('__') else w
                print(f" - {disp} (score: {s:.4f})")

    # Ask user if prediction was correct
    correct_response = input("Was this classification correct? (y = yes / n = no / s = skip): ").lower().strip()
    if correct_response == 'y':
        correct_count += 1
        uc = 'yes'
    elif correct_response == 'n':
        incorrect_count += 1
        uc = 'no'
    else:
        skipped_count += 1
        uc = 'skip'

    # Ask user for true label optionally
    true_label = input("If you know the true label enter 0 (ham) or 1 (spam), or press Enter to skip: ").strip()
    true_label_val = true_label if true_label in ['0', '1'] else ''

    # Append to CSV
    row = {"comment": user_comment, "predicted_class": int(pred), "user_confirmed_correct": uc, "true_label_if_provided": true_label_val}
    pd.DataFrame([row]).to_csv(USER_RESULTS_CSV, mode='a', header=False, index=False)

    print("Saved result to", USER_RESULTS_CSV)
    # update chart counters and display small summary
    total_checked = correct_count + incorrect_count + skipped_count
    print(f"Custom predictions summary — Correct: {correct_count}, Incorrect: {incorrect_count}, Skipped: {skipped_count}, Total checked: {total_checked}")

    # Optionally show updated minimal bar chart of custom counts
    plt.figure(figsize=(4,3))
    plt.bar(['Correct','Incorrect','Skipped'], [correct_count, incorrect_count, skipped_count])
    plt.title("Custom Comments Predictions (Live)")
    plt.ylim(0, max(5, correct_count+incorrect_count+skipped_count + 1))
    plt.show()

print("Interactive session finished. Final counts:")
print(f"Correct: {correct_count}, Incorrect: {incorrect_count}, Skipped: {skipped_count}")
print("All user predictions saved in:", USER_RESULTS_CSV)
