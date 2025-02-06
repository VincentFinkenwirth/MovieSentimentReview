import mlflow
import os

# Save path
SAVE_PATH = "saved_models"

# Models v1:
model_v1 ={
    "rf" : 'runs:/1b78da0367b94391b558810ba17b1660/best_pipeline',
    "nb" : 'runs:/271da4ca0b3d4e35841b59cf2b77f015/best_pipeline',
    "svm" : 'runs:/07c752ee8102464fa82b73571383925c/best_pipeline',
    "logreg" : 'runs:/7e96a95d34094e22af7dad9afe84fc21/best_pipeline',
    "cat" : 'runs:/32ba64e14eb94c278236aca7642a4e75/best_model',
    "bert" : 'runs:/b2f1f4fb5fdf4076a7d0c5a1a97d156b/bert_model'
    }

# Models v2:
model_v2 = {
    "bert" : 'runs:/830d573240b84785bfc8360b0d074a13/bert_model',
    "svm" : 'runs:/6ae2878823714c789e4bffa97a621683/best_pipeline',
    "cat" : 'runs:/7d34cc89d2b4437da68b008c9dd982a4/catboost_model',
    "rf" : 'runs:/4e0f4cddc19c4094ad6648d0bc0d959c/best_pipeline',
    "logreg" : 'runs:/84e6b11d8ceb48ff9168f1fd3ae6189d/best_pipeline',
    "nb" : 'runs:/4564d18622a64c7bbdeaba61929c8db5/best_pipeline'
    }

def save_model(model_name, model_uri, save_path):
    ''' Save single model to disk '''
    # Create directory if not exists
    # Create model path
    model_path = os.path.join(save_path, model_name)
    try:
        if "bert" in model_name:
            model = mlflow.transformers.load_model(model_uri)
            mlflow.transformers.save_model(model, model_path)
        elif "cat" in model_name:
            model = mlflow.catboost.load_model(model_uri)
            mlflow.catboost.save_model(model, model_path)
        else:  # Assume sklearn models
            model = mlflow.sklearn.load_model(model_uri)
            mlflow.sklearn.save_model(model, model_path)


        print(f"Model '{model_name}' saved successfully at {model_path}.")
    except Exception as e:
        print(f"Error saving model '{model_name}': {e}")


def save_models(models, save_path, version):
    ''' Save all models in the provided dictionary. Input: models, save_path, version(1/2) '''
    # Create directory if not exists
    os.makedirs(save_path, exist_ok=True)
    # Save models
    # Version 1
    if version == 1:
        # Create version 1 directory
        os.makedirs(os.path.join(save_path, "v1"), exist_ok=True)
        for model_name, model_uri in models.items():
            save_model(model_name, model_uri, os.path.join(save_path, "v1"))
    # Version 2
    elif version == 2:
        # Create version 2 directory
        os.makedirs(os.path.join(save_path, "v2"), exist_ok=True)
        for model_name, model_uri in models.items():
            save_model(model_name, model_uri, os.path.join(save_path, "v2"))
    else:
        print("Invalid version. Please provide version 1 or 2.")






if __name__ == "__main__":
    # Save models
    save_models(model_v1, SAVE_PATH, version=1)
    save_models(model_v2, SAVE_PATH, version=2)

    print("All models saved successfully.")
