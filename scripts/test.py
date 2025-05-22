import sys
import os

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
import logging
from pathlib import Path

def main():
    load_dotenv(override=True)
    ANNOTATIONS_PATHS = os.getenv("ANNOTATIONS_PATHS").split(',')
    YAML_PATH = os.getenv("YAML_PATH")
    MODEL_TYPE = os.getenv("MODEL_TYPE")
    EXP_NAME = os.getenv("EXP_NAME")
    NEW_ONTOLOGY = os.getenv("NEW_ONTOLOGY").split(',')
    UNWANTED_LABELS = os.getenv("UNWANTED_LABELS").split(',')
    PATH_MODEL_TEST = os.getenv("PATH_MODEL_TEST")
    PATH_STORE_DATA = os.getenv("PATH_STORE_DATA")
    OVERWRITE_DATA = os.getenv("OVERWRITE_DATA", 'False').lower() in ('true', '1', 't')

    TEST_SIZE_str = os.getenv("TEST_SIZE")
    if '.' in TEST_SIZE_str:
        TEST_SIZE = float(TEST_SIZE_str)  # Convert to float if it has a decimal point
    else:
        TEST_SIZE = int(TEST_SIZE_str)  # Keep as int if no decimal point
    
    DESIRED_MARGIN = float(os.getenv("DESIRED_MARGIN"))
    REDUCIBLE_CLASSES = os.getenv("REDUCIBLE_CLASSES")
    TARGET_COUNT = os.getenv("TARGET_COUNT")

    if len(NEW_ONTOLOGY) == 1:
        if NEW_ONTOLOGY[0] == '':
            NEW_ONTOLOGY = None
    
    if len(UNWANTED_LABELS) == 1:
        if UNWANTED_LABELS[0] == '':
            UNWANTED_LABELS = None

    if REDUCIBLE_CLASSES == "" or TARGET_COUNT == "":
        REDUCIBLE_CLASSES = None
        TARGET_COUNT = None
    sr = 32_000
    ecoss_list = []
    yaml_content = load_yaml(YAML_PATH)

    if (Path(PATH_STORE_DATA) / "train").exists() and (Path(PATH_STORE_DATA) / "test").exists() and not OVERWRITE_DATA:
        data_already_generated = True
        logging.warning(f"YOU ARE USING THE DATA STORED IN {PATH_STORE_DATA}")
        signals,labels,split_info = None, None, None
        data_path = PATH_STORE_DATA
    else:
        data_already_generated = False
        for annot_path in ANNOTATIONS_PATHS:
            print(annot_path)
            ecoss_data1 = EcossDataset(annot_path, 'data/', 'zeros', sr, 1,"wav", yaml_content["duration"], True)
            ecoss_data1.add_file_column()
            ecoss_data1.fix_onthology(labels=NEW_ONTOLOGY)
            ecoss_data1.filter_overlapping()
            ecoss_data1.drop_unwanted_labels(UNWANTED_LABELS)
            ecoss_list.append(ecoss_data1)

        ecoss_data = EcossDataset.concatenate_ecossdataset(ecoss_list)
        length_prior_filter = len(ecoss_data.df)
        ecoss_data.filter_lower_sr()
        #ecoss_data.generate_insights()
        if REDUCIBLE_CLASSES and TARGET_COUNT:
            signals,labels,split_info = None, None, None
        else:
            ecoss_data.split_train_test_balanced(test_size=TEST_SIZE, random_state=27)
            if MODEL_TYPE.lower() == "vggish":
                signals,labels,split_info = ecoss_data.process_all_data()
            else:
                _,_,_ = ecoss_data.process_all_data()  # Avoid loading everything in memory for effat and passt

        data_path = ecoss_data.path_store_data

    
    
    results_folder = create_exp_dir(name = EXP_NAME, model=MODEL_TYPE, task= "test")

    if data_already_generated:
        num_classes = len(list((Path(data_path) / "test" ).glob("*")))
    else:
        num_classes = len(ecoss_data.df["final_source"].unique())
    logging.info(f"THE NUMBER OF CLASSES IS {num_classes}\n")
    

    if MODEL_TYPE.lower() == "passt":
        model = PasstModel(yaml_content=yaml_content,data_path=data_path)
    elif MODEL_TYPE.lower() == "effat":
        model = EffAtModel(yaml_content=yaml_content,data_path=data_path, num_classes=10)
    elif MODEL_TYPE.lower() == "vggish":
        model = VggishModel(yaml_content=yaml_content,data_path=data_path, signals=signals, labels=labels, split_info=split_info)

    #model.plot_processed_data()
    model.test(results_folder=results_folder, path_model=PATH_MODEL_TEST, path_data=data_path)

if __name__ == "__main__":
    main()


