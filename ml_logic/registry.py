import time
import os
import pickle
import glob
from tensorflow.keras.models import load_model
from utils.params import *

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics on Google Drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save params locally
    if params is not None:
        #we will create a directory for params
        params_path1 = os.path.join(DRIVE_FOLDER_PATH, "params")
        isExist = os.path.exists(params_path1)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(params_path1)

        params_path = os.path.join(DRIVE_FOLDER_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics locally
    if metrics is not None:
        #we will create a directory for metrics
        metrics_path1 = os.path.join(DRIVE_FOLDER_PATH, "metrics")
        isExist = os.path.exists(metrics_path1)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(metrics_path1)

        metrics_path = os.path.join(DRIVE_FOLDER_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved on drive")


def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it on your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on mlflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    #we will create a directory of models
    model_path1 = os.path.join(DRIVE_FOLDER_PATH, "models")
    isExist = os.path.exists(model_path1)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(model_path1)

    model_path = os.path.join(DRIVE_FOLDER_PATH, "models", f"{timestamp}.h5")
    model.save(model_path)
    print("✅ Model saved on drive")
    return None


#And now can also load model if needed??
def load_model(MODEL_TARGET = "drive") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model found

    """
    #print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)
    # Get latest model version name by timestamp on disk
    local_model_directory = os.path.join(DRIVE_FOLDER_PATH, "models")
    local_model_paths = glob.glob(f"{local_model_directory}/*")
    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
    #print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)
    latest_model = load_model(most_recent_model_path_on_disk)
    print("✅ model loaded from google drive")

    return latest_model
