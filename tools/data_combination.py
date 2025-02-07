# File contains functions for data formatting, dataset combination,
# train/validation/test splitting
# and loading pre-split data for model training and evaluation.

import pandas as pd
from tools.data_preprocess import ClassicPreprocessorSpacy
import joblib as jl
from sklearn.model_selection import train_test_split

#####################################PATHS#####################################
# Dataset 1
IMDB_PATH = "../data/imdb_data.csv"
ROTTEN_PATH = "../data/predictions/rotten_tomatoes_movie_reviews.csv"

#####################################FORMATTING#####################################


# Format functions to rename columns and turn sentiment into binary labels
def rotten_format(data):
    """
    Format Rotten Tomatoes data by renaming columns and converting sentiment
    labels to binary (1 for positive, 0 for negative).

    Parameters:
        data (pd.DataFrame): Raw Rotten Tomatoes data.

    Returns:
        pd.DataFrame: Formatted dataframe with columns ["id", "text", "label"].
    """

    # Select relevant columns
    data = data[["id", "reviewText", "scoreSentiment"]]
    # delete rows with missing values
    data = data.dropna()
    # Rename columns
    data = data.rename(columns={"reviewText": "text", "scoreSentiment": "label"})
    # Turn sentiment into binary
    data["label"] = data["label"].apply(lambda x: 1 if x == "POSITIVE" else 0)
    return data


def imdb_format(data):
    """
    Format IMDb data by renaming columns and converting sentiment
    labels to binary (1 for positive, 0 for negative).

    Parameters:
        data (pd.DataFrame): Raw IMDb data.

    Returns:
        pd.DataFrame: Formatted dataframe with columns ["text", "label"].
    """

    # Select relevant columns
    data = data[["review", "sentiment"]]
    # delete rows with missing values
    data = data.dropna()
    # Rename columns
    data = data.rename(columns={"review": "text", "sentiment": "label"})
    # Turn sentiment into binary
    data["label"] = data["label"].apply(lambda x: 1 if x == "positive" else 0)
    return data


#####################################COMBINING#####################################
# After formatting the rotten data: Extract additional training set equal in size to imdb set
# -> keep remaining data for final testing


def split_rotten_tomatoes(data, train_size=50000, random_seed=42):
    """
    Split the Rotten Tomatoes data into two sets: a subset of a specified size
    for training (balanced between positive and negative) and a remaining subset
    for final testing.

    Parameters:
        data (pd.DataFrame): The formatted Rotten Tomatoes data.
        train_size (int): Total number of samples to include in the training set.
        random_seed (int): Seed for reproducibility of sampling.

    Returns:
        pd.DataFrame: A balanced training subset of size `train_size`.
        pd.DataFrame: The remaining subset of the original data.
    """
    # Split data into positive and negative training data
    class_size = train_size // 2
    positive = data[data["label"] == 1].sample(n=class_size, random_state=random_seed)
    negative = data[data["label"] == 0].sample(n=class_size, random_state=random_seed)
    # Combine the two classes
    train_data = pd.concat([positive, negative])
    # Drop the training data from the original data
    data = data.drop(train_data.index)
    return train_data, data


def combine_datasets(imdb_data, rotten_data):
    """
    Combine IMDb and Rotten Tomatoes data into a single dataframe.

    Parameters:
        imdb_data (pd.DataFrame): IMDb dataset with columns ["text", "label", "token"].
        rotten_data (pd.DataFrame): Rotten dataset with columns ["text", "label", "token"].

    Returns:
        pd.DataFrame: Combined dataframe with columns ["text", "label", "token"].
    """
    # remove id column from rotten data
    rotten_data = rotten_data[["text", "label", "token"]]
    return pd.concat([imdb_data, rotten_data])


def combine_rotten_imdb_training(
        imdb_path, rotten_path,
        save_path_train="../data/preprocessed",
        save_path_rotten="../data/preprocessed"):
    """
    Preprocess and combine IMDb and Rotten Tomatoes data, then save the combined
    training set and the remaining test set.

    1) Reads and formats IMDb and Rotten Tomatoes data.
    2) Transforms each dataset into tokens using the spaCy-based preprocessor.
    3) Splits the Rotten Tomatoes data into training and  final test sets.
    4) Combines the IMDb data with the Rotten Tomatoes training subset.
    5) Saves the combined dataset and the leftover Rotten Tomatoes test data.
    6) Dumps the preprocessor for future reuse.

    Parameters:
        imdb_path (str): Path to the IMDb CSV file.
        rotten_path (str): Path to the Rotten Tomatoes CSV file.
        save_path_train (str): Directory to save the combined training data.
        save_path_rotten (str): Directory to save the leftover Rotten Tomatoes data.
    """
    # Load data
    imdb_data = pd.read_csv(imdb_path)
    rotten_data = pd.read_csv(rotten_path)
    # Format data for consistency
    imdb_data = imdb_format(imdb_data)
    rotten_data = rotten_format(rotten_data)
    # Create preprocessor that returns tokens as string
    preprocessor = ClassicPreprocessorSpacy(return_string=True)
    imdb_data["token"] = preprocessor.transform(imdb_data["text"])
    rotten_data["token"] = preprocessor.transform(rotten_data["text"])

    # Drop empty token lists
    imdb_data = imdb_data[imdb_data["token"] != ""]
    imdb_data = imdb_data.dropna()
    rotten_data = rotten_data[rotten_data["token"] != ""]
    rotten_data = rotten_data.dropna()
    # Split rotten data
    train_data, test_data = split_rotten_tomatoes(rotten_data)
    # Combine datasets
    combined_data = combine_datasets(imdb_data, train_data)
    # Save data
    combined_data.to_csv(f"{save_path_train}/combined_train.csv", index=False)
    test_data.to_csv(f"{save_path_rotten}/rotten_test.csv", index=False)

    # Save preprocessor
    jl.dump(preprocessor, "preprocessor.pkl")


#####################################SPLITTING#####################################
# Function to save train, val and test split


def save_train_val_test(train_data, val_data, test_data, save_path="../data/preprocessed/split_data"):
    """
    Save training, validation, and test splits to CSV files.

    Parameters:
        train_data (pd.DataFrame): Training data.
        val_data (pd.DataFrame): Validation data.
        test_data (pd.DataFrame): Test data.
        save_path (str): Directory where split files will be saved.
    """
    train_data.to_csv(f"{save_path}/train.csv", index=False)
    val_data.to_csv(f"{save_path}/val.csv", index=False)
    test_data.to_csv(f"{save_path}/test.csv", index=False)


# Function to create train, val and test split using stratify to keep distribution
def create_train_val_test(data, test_size=0.2, save_path="../data/preprocessed/split_data"):
    """
    Create training, validation, and test splits using stratification,
    then save them to disk.

    1) Splits the data into train and a temporary set.
    2) Splits the temporary set into validation and test sets.
    3) Saves each split.

    Parameters:
        data (pd.DataFrame): The dataset containing ["text", "label", "token"].
        test_size (float): Fraction of data to withhold for the temporary set
                           (and later equally split into val/test).
        save_path (str): Directory where split files will be saved.
    """
    # Split data into train and test
    train_data, temp_data = train_test_split(data, test_size=test_size, stratify=data["label"], random_state=42)
    # Split train data into train and validation
    val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data["label"], random_state=42)
    save_train_val_test(train_data, val_data, test_data, save_path)


######################DATA_V1######################
def create_data_v1(data_path, save_path):
    """
    Prepare 'version 1' data from a given CSV. This involves:
    1) Reading and formatting IMDb data.
    2) Tokenizing data using the ClassicPreprocessorSpacy.
    3) Creating and saving train/validation/test splits.

    Parameters:
        data_path (str): Path to the CSV file containing IMDb data.
        save_path (str): Directory to save the split data.
    """
    # Load data
    data = pd.read_csv(data_path)
    # Format data
    data = imdb_format(data)
    # Create preprocessor
    preprocessor = ClassicPreprocessorSpacy(return_string=True)
    data["token"] = preprocessor.transform(data["text"])
    # Create splits
    create_train_val_test(data, save_path=save_path)


#####################################LOADING#####################################
# Loads data and returns X_train, X_test, X_val, y_train, y_test, y_val
def load_data(path, val_set=True, X_col="token"):
    """
    Load pre-split data from the specified directory and return
    train/(val)/test features and labels.

    Parameters:
        path (str): Directory containing 'train.csv', 'val.csv', and 'test.csv'.
        val_set (bool): Whether to return a validation set. If False, validation
                        data is merged with train.
        X_col (str): Name of the feature column.

    Returns:
        If val_set is True:
            X_train, X_val, X_test, y_train, y_val, y_test
        Otherwise:
            X_train, X_test, y_train, y_test
    """
    train_data = pd.read_csv(f"{path}/train.csv")
    test_data = pd.read_csv(f"{path}/test.csv")
    val_data = pd.read_csv(f"{path}/val.csv")
    # Combine train and val data if val_set is False for cases where cross-val is used
    if not val_set:
        train_data = pd.concat([train_data, val_data])
    # Retrieve X and y
    X_train = train_data[X_col]
    X_test = test_data[X_col]
    y_train = train_data["label"]
    y_test = test_data["label"]
    if val_set:
        X_val = val_data[X_col]
        y_val = val_data["label"]
        return X_train, X_val, X_test, y_train, y_val, y_test
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    #combine_rotten_imdb_training(IMDB_PATH, ROTTEN_PATH)
    #create_train_val_test(pd.read_csv("../data/preprocessed/combined_train.csv"))
    pass








