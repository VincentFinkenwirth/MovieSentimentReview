# File contains the Preprocessor class that uses spaCy to perform common text-cleaning tasks,
# including optional removal of stopwords and HTML tags, lowercasing, and token lemmatization.
# The output can be returned as a list of tokens or as a single string of tokens.
import pandas as pd
import re
import spacy


class ClassicPreprocessorSpacy:
    """
    A preprocessor class that uses spaCy to perform common text-cleaning tasks,
    including optional removal of stopwords and HTML tags, lowercasing, and
    token lemmatization. The output can be returned as a list of tokens or as
    a single string of tokens.
    """
    def __init__(self, batch_size=1000, n_process=2, remove_stopwords=True, lower=True, return_string=False):
        """
        Initialize the preprocessor with spaCy and various configuration options.

        Parameters:
            batch_size (int): Number of texts to process in each batch.
            n_process (int): Number of parallel processes to use in spaCy's pipe.
            remove_stopwords (bool): Whether to remove stopwords from tokens.
            lower (bool): Whether to convert text to lowercase.
            return_string (bool): If True, returns the tokens as a single string.
        """
        # Load the spaCy model with unnecessary components disabled for efficiency
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.html_tag_re = re.compile(r"<.*?>")
        self.batch_size = batch_size
        self.n_process = n_process
        self.remove_stopwords = remove_stopwords
        self.lower = lower
        self.return_string = return_string

    def transform(self, df):
        """
        Clean, tokenize, and optionally remove stopwords and lowercase text based on how the
        preprocessor was configured during initialization. The steps performed are as follows:

        1) Renames the first column to "text" if df is a DataFrame.
        2) Removes HTML tags.
        3) Converts text to lowercase (if enabled).
        4) Uses spaCy to tokenize and lemmatize the text.
        5) Optionally removes stopwords and non-alphabetic tokens.
        6) Returns tokens as a list or as a space-joined string.

        Parameters:
            df (pd.DataFrame or pd.Series): Input text data. If a Series is provided,
                                            it will be converted to a DataFrame.

        Returns:
            pd.Series: A series containing the transformed tokens
                       (either list of tokens or space-joined string).
        """
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



