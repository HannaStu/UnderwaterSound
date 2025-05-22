import sys
import os

from pathlib import Path

PROJECT_FOLDER = os.path.dirname(__file__).replace('/src', '/pipeline')
PARENT_PROJECT_FOLDER = os.path.dirname(PROJECT_FOLDER)
sys.path.append(PARENT_PROJECT_FOLDER)
src_dir = os.path.join(PARENT_PROJECT_FOLDER, 'src')
sys.path.append(src_dir)

from src.pipeline.passt_model import PasstModel
from src.pipeline.effat_model import EffAtModel
from src.pipeline.vggish_model import VggishModel
from src.pipeline.dataset import EcossDataset
from src.pipeline.utils import create_exp_dir, load_yaml
from dotenv import load_dotenv
# import torch
import logging
import shutil



def main():
    # torch.set_float32_matmul_precision('high') #In a torch warning says you should run in this mode. I am not sure about the implications
    load_dotenv(override=True)
    ANNOTATIONS_PATHS = os.getenv("ANNOTATIONS_PATHS").split(',')
    YAML_PATH = os.getenv("YAML_PATH")
    MODEL_TYPE = os.getenv("MODEL_TYPE")
    EXP_NAME = os.getenv("EXP_NAME")
    NEW_ONTOLOGY = os.getenv("NEW_ONTOLOGY").split(',')
    UNWANTED_LABELS = os.getenv("UNWANTED_LABELS").split(',')
    TEST_SIZE = float(os.getenv("TEST_SIZE"))
    DESIRED_MARGIN = float(os.getenv("DESIRED_MARGIN"))
    REDUCIBLE_CLASSES = os.getenv("REDUCIBLE_CLASSES").split(',')
    TARGET_COUNT = os.getenv("TARGET_COUNT").split(',')
    PATH_STORE_DATA = os.getenv("PATH_STORE_DATA")
    PAD_MODE = os.getenv("PAD_MODE")
    OVERWRITE_DATA = os.getenv("OVERWRITE_DATA", 'False').lower() in ('true', '1', 't')
    



    MIN_DURATION = float(os.getenv("MIN_DURATION"))
    CLASSES_FILTER_TIME = os.getenv("CLASSES_FILTER_TIME").split(',')
    if CLASSES_FILTER_TIME[0] == "":
        CLASSES_FILTER_TIME = None

    if len(NEW_ONTOLOGY) == 1:
        if NEW_ONTOLOGY[0] == '':
            NEW_ONTOLOGY = None
    
    if len(UNWANTED_LABELS) == 1:
        if UNWANTED_LABELS[0] == '':
            UNWANTED_LABELS = None

    if REDUCIBLE_CLASSES == [""] or TARGET_COUNT == [""]:
        REDUCIBLE_CLASSES = None
        TARGET_COUNT = None

    ecoss_list = []
    yaml_content = load_yaml(YAML_PATH)
    if MODEL_TYPE.lower() == "vggish":
        duration = 1
        sr = yaml_content["desired_sr"]
    else:
        duration = yaml_content["duration"]
        sr = yaml_content["sr"]

    if (Path(PATH_STORE_DATA) / "train").exists() and (Path(PATH_STORE_DATA) / "test").exists() and not OVERWRITE_DATA:
        data_already_generated = True
        logging.warning(f"YOU ARE USING THE DATA STORED IN {PATH_STORE_DATA}")
        signals,labels,split_info = None, None, None
        data_path = PATH_STORE_DATA
    else:
        data_already_generated = False
        if PATH_STORE_DATA and (Path(PATH_STORE_DATA) / "train").exists() and (Path(PATH_STORE_DATA) / "test").exists():
            shutil.rmtree(Path(PATH_STORE_DATA) / "train")
            shutil.rmtree(Path(PATH_STORE_DATA) / "test")


        for annot_path in ANNOTATIONS_PATHS:
            logging.info(annot_path)
            ecoss_data1 = EcossDataset(annot_path, PATH_STORE_DATA, PAD_MODE, sr, duration, "wav", DESIRED_MARGIN, yaml_content["window"])
            ecoss_data1.add_file_column()
            ecoss_data1.fix_onthology(labels=NEW_ONTOLOGY)
            ecoss_data1.filter_overlapping()
            ecoss_data1.drop_unwanted_labels(UNWANTED_LABELS)
            ecoss_list.append(ecoss_data1)
        ecoss_data = EcossDataset.concatenate_ecossdataset(ecoss_list)
        length_prior_filter = len(ecoss_data.df)
        ecoss_data.generate_insights()
        ecoss_data.filter_lower_sr()
        logging.info(f"\nThe number of signals after filtering by sr is {len(ecoss_data.df)}\n")
        ecoss_data.filter_by_duration(min_duration=MIN_DURATION, set_classes=CLASSES_FILTER_TIME)
        logging.info(f"\nThe number of signals after filtering by duration is {len(ecoss_data.df)}\n")
        ecoss_data.filter_by_freqlims()
        logging.info(f"\nThe number of signals after filtering by frequency limits is {len(ecoss_data.df)}\n")
        if MODEL_TYPE.lower() == "vggish":
                if TARGET_COUNT:
                    ecoss_data.filter_amount(reducible_classes = REDUCIBLE_CLASSES, target_count = [int(item) for item in TARGET_COUNT])

        ecoss_data.generate_insights()
        ecoss_data.split_train_test_balanced(test_size=TEST_SIZE, random_state=27)
        if MODEL_TYPE.lower() == "vggish":
            signals,labels,split_info = ecoss_data.process_all_data()
        else:
            _,_,_ = ecoss_data.process_all_data()  # Avoid loading everything in memory for effat and passt
        data_path = ecoss_data.path_store_data


    results_folder = create_exp_dir(name = EXP_NAME, model=MODEL_TYPE, task= "train")
    if data_already_generated:
        num_classes = len(list((Path(data_path) / "train" ).glob("*")))
    else:
        num_classes = len(ecoss_data.df["final_source"].unique())
    logging.info(f"THE NUMBER OF CLASSES IS {num_classes}\n")
    if data_already_generated:
        num_classes = len(list((Path(data_path) / "train" ).glob("*")))
    else:
        num_classes = len(ecoss_data.df["final_source"].unique())
    logging.info(f"THE NUMBER OF CLASSES IS {num_classes}\n")

    if MODEL_TYPE.lower() == "passt":
        model = PasstModel(yaml_content=yaml_content,data_path=data_path)
    elif MODEL_TYPE.lower() == "effat":
        model = EffAtModel(yaml_content=yaml_content,data_path=data_path, num_classes=num_classes)
    elif MODEL_TYPE.lower() == "vggish":
        model = VggishModel(yaml_content=yaml_content,data_path=data_path, signals=signals, labels=labels, split_info=split_info)


    model.plot_processed_data()
    model.train(results_folder = results_folder)

if __name__ == "__main__":
    main()
