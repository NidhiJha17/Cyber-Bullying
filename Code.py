
#Data Loading
import pandas as pd
datapath = r"C:\Users\nidhi\Downloads\cyber bullying dataset\aggression_parsed_dataset.csv"
df = pd.read_csv(datapath)

#keeping only the columns that we need
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