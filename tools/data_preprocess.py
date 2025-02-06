import pandas as pd
import re
import spacy


class ClassicPreprocessorSpacy:
    def __init__(self, batch_size=1000, n_process=2, remove_stopwords=True, lower=True, return_string=False):
        # Load the spaCy model with unnecessary components disabled for efficiency
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.html_tag_re = re.compile(r"<.*?>")
        self.batch_size = batch_size
        self.n_process = n_process
        self.remove_stopwords = remove_stopwords
        self.lower = lower
        self.return_string = return_string

    def transform(self, df):
        # Ensure input is a DataFrame with a 'text' column
        if isinstance(df, pd.DataFrame):
            df = df.rename(columns={df.columns[0]: "text"})
        else:
            df = pd.DataFrame(df, columns=["text"])

        # Remove HTML tags
        df["text"] = df["text"].str.replace(self.html_tag_re, "", regex=True)

        # Convert text to lowercase if specified
        if self.lower:
            df["text"] = df["text"].str.lower()

        # Process texts with spaCy's pipeline
        docs = list(self.nlp.pipe(df["text"], batch_size=self.batch_size, n_process=self.n_process))

        # Extract lemmatized tokens and remove non-alphabetic tokens
        # excluding stop words
        if self.remove_stopwords:
            df["tokens"] = [
                [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
                for doc in docs]
        else: # Keep stopwords
            df["tokens"] = [
                [token.lemma_ for token in doc if token.is_alpha]
                for doc in docs]
        # Turn list of tokens into string

        if self.return_string:
            df["tokens"] = df["tokens"].apply(lambda x: " ".join(x))

        return df["tokens"]



