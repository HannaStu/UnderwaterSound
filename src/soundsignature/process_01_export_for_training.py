"""
Copyright (C) 2024 Witteveen+Bos (https://www.witteveenbos.com)

Licensed under the EUPL, Version 1.2 or - as soon they will be approved by
the European Commission - subsequent versions of the EUPL (the "Licence");
You may not use this work except in compliance with the Licence.
You may obtain a copy of the Licence at:
https://joinup.ec.europa.eu/software/page/eupl

Unless required by applicable law or agreed to in writing, software
distributed under the Licence is distributed on an "AS IS" basis,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the Licence for the specific language governing permissions and
limitations under the Licence.
"""
# Internal modules
import datetime as dt
import logging
import os
import sys

# External modules
import dotenv
import pandas as pd

PARENT_PROJECT_FOLDER = r"C:\Users\hurh\GitHub\underwater-noise\src"
sys.path.append(PARENT_PROJECT_FOLDER)

# And helperfunctions
import soundsignature.log as log
import soundsignature.utilities as util
from soundsignature.annotation import ANNOTATION_FILES

# Importing own classes
from soundsignature.soundsignaldataset import SoundSignalDataset

# Load the data and labelled set
dotenv.load_dotenv(".env")
util.check_paths()

# Path definitions
# Inputs: data, meta data and labels
metadata_path = os.path.join(os.getenv("DATADIR"), os.getenv("METADATA"))
# Loading the dataset
dataset_id = int(os.getenv("DATASET_ID"))
dataset = SoundSignalDataset.from_excel(dataset_id=dataset_id, path_excel=metadata_path)
dataset_path = os.path.join(os.getenv("DATADIR"), dataset.folder)

# get the label paths of the dataset
label_paths = ANNOTATION_FILES[dataset.folder]
# get the full path of the label paths
label_paths = [os.path.join(os.getenv("DATADIR"), label_path) for label_path in label_paths]
# get the process dictionary path
process_dict_path = os.path.join(os.getenv("DATADIR"), "process_dict.xlsx")

# Outputs:
log_path = os.path.join(os.getenv("EXPORTDIR"), "logging")
os.makedirs(log_path, exist_ok=True)
export_path = os.path.join(os.getenv("EXPORTDIR"), dataset.folder, "samples for training")
os.makedirs(export_path, exist_ok=True)


# Configuring logging: one logger, with two handles (to both screen and file)
log_file = log.defLogFile(logdir=log_path, prefix=f"export-for-training-{dataset.folder}-")
log.startLogging(log_file, level=logging.INFO, header="Sampling Dataset for Training")

# check on existence of all paths
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path [{dataset_path}] not found")
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"Metadata path [{metadata_path}] not found")
for label_path in label_paths:
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"File with annotations [{label_path}] not found")
if not os.path.exists(process_dict_path):
    raise FileNotFoundError(f"Process dictionary [{process_dict_path}] not found")

logging.info("Loading dataset and labels...")

# get all files with labels
files_with_labels = []
for label_path in label_paths:
    labels = pd.read_excel(label_path)
    files_with_labels += list(labels["Audio Name"].unique())

# get basename of all files_with_labels
files_with_labels = [os.path.basename(file) for file in files_with_labels]
# get extension of all files_with_labels
list_with_extensions = list(set([os.path.splitext(file)[1] for file in files_with_labels]))

# Append the files from the dataset
if len(list_with_extensions) == 1:
    dataset.append_files_from_folder(
        list_of_files=files_with_labels, extension=list_with_extensions[0]
    )
else:
    dataset.append_files_from_folder(list_of_files=files_with_labels)

# Append the labels from all label paths
for label_path in label_paths:
    dataset.append_labels(process_dict_path=process_dict_path, path_label_excel=label_path)

# Export files for training
dataset.export_files_with_annotations(export_path)

# # get list of all annotations
# all_annotations = []
# for file in dataset.files:
#     all_annotations += file.annotations

# # simplify, only export first 3 samples
# all_annotations = all_annotations[:3]

# # for this dataset, export samples with metadata csv
# dataset.export_samples_for_training(
#     path_export=os.path.join(dataset_path, "samples for training"),
#     annotations=all_annotations,
# )


log.stopLogging("Done with exporting samples for training")
