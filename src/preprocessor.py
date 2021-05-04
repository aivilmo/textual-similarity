import nltk
import pandas as pd
from similarity import Similarity
import csv
from sklearn.feature_extraction.text import TfidfVectorizer


class Preprocessor:
    """ Class to Preprocess data, """

    """ Read file, tokenize, etc  """

    def initialize(self, namefile):
        return pd.read_csv(
            namefile,
            sep="\t",
            encoding="utf-8",
            error_bad_lines=False,
            quoting=csv.QUOTE_NONE,
            names=[
                "genre",
                "set",
                "year",
                "id",
                "similarity",
                "sentence_a",
                "sentence_b",
            ],
        ).dropna()

    def tokenize(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        cleaned_tokens = list(filter(str.isalnum, tokens))
        cleaned_tokens = [w.lower() for w in cleaned_tokens]
        return cleaned_tokens

    def tag(self, sentence):
        tokens = self.tokenize(sentence)
        return nltk.pos_tag(tokens)

    def filter(self, tokens):
        tagged_tokens = nltk.pos_tag(tokens)
        filtered_tokens = []
        for tagged_token in tagged_tokens:
            token = tagged_token[0]
            tag = tagged_token[1]
            if "VB" in tag:
                filtered_tokens.append((token, "v"))
            if "NN" in tag:
                filtered_tokens.append((token, "n"))
        return filtered_tokens

    def tokenize_and_filter_dataset(self, df):
        df_filtered = df.copy()
        df_filtered["sentence_a"] = df_filtered.sentence_a.apply(self.tokenize).apply(
            self.filter
        )
        df_filtered["sentence_b"] = df_filtered.sentence_b.apply(self.tokenize).apply(
            self.filter
        )
        return df_filtered

    def tag_dataset(self, df):
        df["sentence_a"] = df.sentence_a.apply(self.tag)
        df["sentence_b"] = df.sentence_b.apply(self.tag)
        return df

    def idf_from_dataset(self, df):
        vectorizer = TfidfVectorizer(use_idf=True)
        X = vectorizer.fit_transform(df["sentence_a"] + df["sentence_b"])
        idf = vectorizer.idf_
        print("Calculating index idf from dataframe")
        return dict(zip(vectorizer.get_feature_names(), idf))
