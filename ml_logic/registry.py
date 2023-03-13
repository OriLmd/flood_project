# External imports
import glob
import os
import time
import csv
from tensorflow.keras import models

# Internal imports
from ml_logic import metrics

def save_dict_csv(csv_path, dict_to_save):
        # Open the CSV file for writing
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # Write the header row
            writer.writerow(['key', 'value'])

            # Write each key-value pair in the dictionary to a new row in the CSV file
            for key, value in dict_to_save.items():
                writer.writerow([key, value])
        return None

def save_results(params, metrics, drive_folder_path, model_name = 'unet'):
    """
    params = history.params & metrics= history.history
    Persist params & metrics locally on hard drive at
    "{drive_folder_path}/params/{model_name}_{current_timestamp}_params.csv"
    "{drive_folder_path}/metrics/{model_name}_{current_timestamp}_metrics.csv"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save params
    if params is not None:
        params_path = os.path.join(drive_folder_path, "params", f"{model_name}_{timestamp}_params" + ".csv")
        save_dict_csv(params_path, params)
        print("✅ Params saved")

    # save metrics
    if metrics is not None:
        metrics_path = os.path.join(drive_folder_path, "metrics", f"{model_name}_{timestamp}_metrics" + ".csv")
        save_dict_csv(metrics_path, metrics)
        print("✅ Metrics saved")

    return None

def save_evaluation(eval_dict, drive_folder_path, model_name = 'unet'):
    """
    eval_dict = model_unet.evaluate(test_dataset_preproc.batch(16), return_dict=True)
    Persist evaluation metrics locally on drive at
    "{drive_folder_path}/metrics/{model_name}_{current_timestamp}_metrics_eval.csv"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save metrics
    if eval_dict is not None:
        evals_path = os.path.join(drive_folder_path, "metrics", f"{model_name}_{timestamp}_metrics_eval" + ".csv")
        save_dict_csv(evals_path, eval_dict)
        print("✅ Evaluation metrics saved")

    return None

def save_model(model_to_save, drive_folder_path, model_name = 'unet'):
    """
    Persist trained model locally on hard drive at f"{drive_folder_path}/models/{model_name}_{current_timestamp}.h5"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save model locally
    model_path = os.path.join(drive_folder_path, "models", f"{model_name}_{timestamp}.h5")
    model_to_save.save(model_path)

    print("✅ Model saved locally")
    return None


def load_saved_model(drive_folder_path, custom={"DiceLoss": metrics.DiceLoss(), "Dice":metrics.Dice(), 'TotalError':metrics.TotalError()}):
    """
    custom is a dictionnary with the custom objects used in the model to be loaded, i.e. loss, metrics...
    Return a saved model:
    - from drive (latest one in alphabetical order)

    Return None (but do not Raise) if no model found
    """

    # Get latest model version name by timestamp on disk

    drive_model_directory = os.path.join(drive_folder_path, "models")
    drive_model_paths = glob.glob(f"{drive_model_directory}/*")
    if not drive_model_directory:
        print("✅ No model find")
        return None

    most_recent_model_path_on_disk = sorted(drive_model_paths)[-1]
    print(f"\nLoad latest model from disk...")
    lastest_model = models.load_model(most_recent_model_path_on_disk, custom_objects=custom)
    print("✅ model loaded from local disk")

    return lastest_model
