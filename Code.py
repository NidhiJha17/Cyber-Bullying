
#Data Loading
import pandas as pd
datapath = r"C:\Users\nidhi\Downloads\cyber bullying dataset\aggression_parsed_dataset.csv"
df = pd.read_csv(datapath)

#1.) Keeping only the columns that we need
df = df[["Text", "oh_label"]]

#remove any empty rows
df = df.dropna().reset_index(drop=True)

#Exploratory Data Analysis EDA

import matplotlib.pyplot as plt

#Plot-1:Class Distribution

labels = df["oh_label"].map({0: "Non-Aggressive", 1: "Aggressive"})
labels.value_counts().plot(kind="bar", color=["steelblue", "tomato"], edgecolor="white")
plt.title("Class Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plot1_class_distribution.png")
plt.show()

#Plot-2: Text Length Distribution

df["text_length"] = df["Text"].apply(len)
df.groupby(df["oh_label"].map({0: "Non-Aggressive", 1: "Aggressive"}))["text_length"].plot(
    kind="hist", bins=50, alpha=0.6, legend=True
)
plt.title("Text Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plot2_text_length.png")
plt.show()

#Plot-3 : Word Clouds

from wordcloud import WordCloud
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (label_id, label_name, cmap) in zip(axes, [
    (0, "Non-Aggressive", "Blues"),
    (1, "Aggressive",     "Reds")
]):
    all_text = " ".join(df[df["oh_label"] == label_id]["Text"].astype(str))
    wc = WordCloud(width=600, height=300, background_color="white",
                   colormap=cmap, stopwords=stop_words).generate(all_text)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(f"Word Cloud — {label_name}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("plot3_wordclouds.png")
plt.show()


#NLP Preproessing for text cleaning
#sTEP:1)- Basic Text cleaning
import re
import string

def clean_text (text):
    text= text.lower()
    text= re.sub(r"https\s+"," ", text)
    text= re.sub(r"[^\w\s]", " ", text)
    text= text.strip()
    return text

df['clean_text']= df['Text'].apply(clean_text)


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('Stopwords')
nltk.download('Wordnet')

#Step:2) TOKENIZATION

df['tokens']= df['Text'].apply(lambda x: x.split())

#Step-3) REMOVING STOPWORDS
stop_words= set(stopwords.words('English'))

df['tokens']=df['tokens'].apply(lambda words: [word for word in words if word not in stop_words])

#Step:4) - Lemmatization (converting  words to its root form)
lemmatizer= WordNetLemmatizer()
df['tokens']= df['tokens'].apply(lambda words: [lemmatizer.lemmatize(word) for word in words])

#Step:5)- Joining words back to the sentence

df['clean_text']= df['tokens']. apply(lambda x: " ". join(x))

#Step:5)- Converting text to numerical features

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer= TfidfVectorizer(max_features=5000)
x= vectorizer.fit_transform(df['clean_text'])
y= df['oh_label']