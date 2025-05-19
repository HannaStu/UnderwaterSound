"""
Created on Fri Aug 02 10:10:48 2024

@authors: Pablo Aguirre, Isabel Carozzo, Jose Antonio García, Mario Vilar
"""

""" This script implements the class VggishModel which is responsible for all the training, testing and inference stuff related with the
    Vggish + SVM model """
    
from src.pipeline.utils import *
import json
from tqdm import tqdm
import os
import time
from sklearn.svm import SVC
import joblib
import logging
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import soundfile as sf
from src.models.VggishFeaturesExtractor import *
import shutil
logging.basicConfig(level=logging.INFO)
from models.vggish_modules import vggish_input
logger = logging.getLogger(__name__)

handler = logging.FileHandler("log.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class VggishModel():
    def __init__(self, yaml_content: dict, signals: list = None, labels:list = None, split_info:list = None, data_path: str = None) -> None:
        self.yaml = yaml_content
        self.sample_rate=self.yaml.get('desired_sr')
        self.signals = signals
        self.labels = labels
        self.split_info = split_info
        self.data_path = data_path
        self.features_extractor = VggishFeaturesExtractor(sample_rate = self.sample_rate)
        self.model  = SVC()


    def train(self, results_folder):
        """
        Compute the training. The data are processed by the feature extractor and the labels are encoded.
        The training is performed using the best parameters found by the gridsearch for the SVM.
        The model and the training plot are saved in the results_folder

        Parameters:
        results_folder (str): path to the folder where the results are saved
        
        Returns:
        None
        """

        #traindata is array of size len(segments)
        #train data[0] is array of (1) array of size 32000 sound (2) string label source
        train_data = self.get_data(X = self.signals, Y = self.labels, split_info = self.split_info, data_path = self.data_path, split="train")
        OVERWRITE_EMBEDDINGS = self.yaml.get('OVERWRITE_EMBEDDINGS')
        embedding_path = self.yaml.get('embedding_path')
        if OVERWRITE_EMBEDDINGS:
            X, Y = self.data_preparation(data = train_data, saving_folder = results_folder)
            if embedding_path:
                embedding_file = Path(embedding_path) / "all_embeddings.npy"
                np.save(embedding_file, np.array(X))  # Save the entire X list as a single .npy file
        else:
            embedding_file = Path(embedding_path) / "all_embeddings.npy"
            if embedding_file.exists():
                # Load the saved embeddings
                X = np.load(embedding_file, allow_pickle=True)
                logger.info(f"Loaded embeddings from {embedding_file}")
                Y = [y for _,y in train_data]
                Y = self.get_labels_encoding(Y, saving_folder = results_folder)
            else:
                logger.error(f"Embeddings file not found at {embedding_file}. Cannot proceed without embeddings.")
                exit()

        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.yaml.get('grid_search_params'),
            error_score='raise',
            scoring='accuracy',
            cv=5,
            return_train_score=True
        )
        grid = grid_search.fit(X, Y)
        best_params = grid.best_params_
        # Set the best hyperparameters and fit the model
        self.model = self.model.set_params(**best_params)
        self.model = self.model.fit(X,Y)
        Y_pred = self.model.predict(X)
        self.save_weigths(saving_folder = results_folder)
        self.plot_results(set = 'train', saving_folder = results_folder, gridsearch = grid, y_true = Y, y_pred = Y_pred)


    def test(self, results_folder, path_model= None, path_data = None):
        """
        Compute the test. The data are processed by the feature extractor and the labels are encoded.
        The metrics "accuracy" and "macro F1" are computed and saved in the results_folder.
        The test plot are saved in results_folder.

        Parameters:
        results_folder (str): path to the folder where the results are saved
        
        Returns:
        None
        """
        test_data = self.get_data(X = self.signals, Y = self.labels, split_info = self.split_info, data_path = self.data_path, split="test")
        X, Y = self.data_preparation(data = test_data, saving_folder = results_folder)
        if self.yaml.get('model_path'):
            self.model = joblib.load(self.yaml.get('model_path'))
        Y_pred = self.model.predict(X)
        Y_prob = self.model.predict_proba(X)
        
        examples_to_plot = 3
        for i in (1000, 1020, 1050,18000,20000,22000):
            x = X[i]
            y = Y[i]
            mel = test_data[i][0]
            self.generate_heatmap_svm(self.model, x, y, mel)




        with open(self.yaml.get('labels_mapping_path'), 'r') as file:
            class_mapping = json.load(file)  
        probs_outputs_per_class = {int(cls): [] for cls in class_mapping.keys()}

        # Iterate over predictions and probabilities
        for prob_vector in Y_prob:
            # Find the class with the highest probability
            highest_prob_class = np.argmax(prob_vector)
            # Append the probability vector to the corresponding class
            probs_outputs_per_class[highest_prob_class].append(prob_vector.tolist())

        #   Compute the average probability vector for each class
        avg_probs_per_class = {
            cls: np.mean(vectors, axis=0).tolist() if len(vectors) > 0 else [0] * len(class_mapping)
            for cls, vectors in probs_outputs_per_class.items()
        }

        # Save the average probabilities per class to a JSON file
        with open(os.path.join(results_folder, 'avg_probs_per_class.json'), 'w') as json_file:
            json.dump(avg_probs_per_class, json_file)
            print(avg_probs_per_class)

        accuracy, f1 = self.compute_scores(Y, Y_pred)
        result = {'accuracy':accuracy, 'f1_score' : f1}
        with open(os.path.join(results_folder, 'metrics.json'), 'w') as json_file:
            json.dump(result, json_file)
        self.plot_results(set = 'test', saving_folder = results_folder, y_true = Y, y_pred = Y_pred)
        

    def inference(self,path_data,results_folder,path_model=None):

        self.results_folder = Path(results_folder)
        duration = self.yaml.get('desired_duration')
        
        segments = process_data_for_inference(path_audio= path_data, desired_sr=self.sample_rate, desired_duration=duration)
        X = []
        start = time.time()
        for segment in tqdm(segments, total = len(segments), desc = "Features extraction"):
            X.append(self.get_features(segment))

        if self.yaml.get('model_path'):
            self.model = joblib.load(self.yaml.get('model_path'))
        else:
            logger.error('Error. model_path missing in the yaml configuration file')
            exit()

        predictions = {}
        predictions["class by time interval"] = {}
        s = 0
        for x in X:
            x = x.reshape(1,-1)
            y = self.model.predict(x)
            labels_mapping_path = self.yaml.get('labels_mapping_path')
            with open(labels_mapping_path, 'r') as file:
                label_mapping = json.load(file)
                y= np.array([label_mapping[str(label)] for label in y])
                predictions["class by time interval"][f'\n{s}-{s+1}'] = y[0]
                s+=1
        total_time = time.time() - start
        logger.info(f"Total time for inference: {total_time} seconds")
        with open(os.path.join(results_folder, 'predictions.json'), "w") as f:
            json.dump(predictions, f)


    def plot_results(self, set, saving_folder, y_true, y_pred, gridsearch = None):
        """
        Compute the plots (Confusion Matrix/Training Curves) based on the procedure(train/test).

        Parameters:
        set (str): 'train' | 'test'
        saving_folder (str): path to the folder where the results are saved
        gridsearch (obj, optional): instance of fitted estimator.Defaults to None.
        y_true (array, optional): ground truth  
        y_pred (array, optional): labels predicted.
        
        Returns:
        None
        """
        if set == 'train':
            cf_matrix = ConfusionMatrix(labels_mapping_path = os.path.join(saving_folder, 'labels_mapping.json'))
            fig1 = cf_matrix.plot(y_true = y_true, y_pred = y_pred)
            cf_matrix.save_plot(plot = fig1, saving_folder = saving_folder)
            
            training_curves = ValidationPlot(gridsearch = gridsearch)
            fig2 = training_curves.plot()
            training_curves.save_plot(plot = fig2, saving_folder = saving_folder)
        else:
            cf_matrix = ConfusionMatrix(labels_mapping_path = os.path.join(saving_folder, 'labels_mapping.json'))
            fig1 = cf_matrix.plot(y_true = y_true, y_pred = y_pred)
            cf_matrix.save_plot(plot = fig1, saving_folder = saving_folder)


    def save_weigths(self,saving_folder):
        """
        Save the weigths of the model

        Parameters:
        saving_folder (str): path to the folder where the results are saved
        
        Returns:
        None
        """
        with open(os.path.join(saving_folder,'model.joblib'), 'wb') as f:
            joblib.dump(self.model, f)


    def plot_processed_data(self):
        path_classes = os.path.join(self.data_path, "train")
        available_classes = os.listdir(path_classes)
        for av_class in available_classes:
            path_wavs = os.path.join(path_classes, av_class)
            wav_to_plot = os.path.join(path_wavs,np.random.choice(os.listdir(path_wavs)))
            logger.info(f"The file that will be plotted is {wav_to_plot}")

            #y, sr = torchaudio.load(wav_to_plot)
            melspec = vggish_input.wavfile_to_examples(wav_to_plot, self.sample_rate)
            logger.info(f"The shape of the melspec is {melspec.shape}")

            plt.figure()
            plt.imshow(melspec[0], origin="lower")
            plt.title(av_class)
            plt.show()
            plt.close()
    
    def generate_heatmap_svm(model, features, pred_label, mel_spec):
        """
        Generates a heatmap showing which parts of the Mel-spectrogram are important for the SVM decision.
    
        svm_model: Trained SVM model.
        features: The VGGish embeddings (output from the VGGish model).
        mel_spectrogram: The Mel spectrogram image.
        target_class_index: The target class index for which the heatmap is generated.
    
        Returns: The heatmap for the target class.
        """
        # Extract the weight vector (coefficients) for the target class
        weights = model.coef_[pred_label]  # Shape: (n_features,)
    
        # Normalize the feature importance (weights) to match the size of the spectrogram
        feature_importance = np.abs(weights)  # Get absolute weights as feature importance

        # Normalize the importance to the range [0, 1]
        feature_importance /= np.max(feature_importance)

        # Now we need to map this feature importance back to the Mel spectrogram
        # In this case, we assume that the embeddings are a flattened version of the spectrogram.
        # The number of features (input size to the SVM) should match the flattened mel spectrogram's size.

        # Reshape the importance to match the spectrogram's dimensions
        heatmap = np.reshape(feature_importance, mel_spec.shape)  # Reshape to spectrogram size
        
        # Normalize the heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap





    def get_features(self, x):
        """
        Data preprocessing: 
        - extraction of features using VGGish feature extractor 
        - flatten the features extracted, resulting in a one-dimensional vector

        Parameters:
        x (array): signal

        Returns:
        array
        """
        vggish_features = self.features_extractor(x)
        return flatten(vggish_features)
    
    
    def get_labels_encoding(self, Y, saving_folder):
        """
        Labels encoding. 
        If a labels encoding is provided it is used, otherwhise it is created.
        The labels enconding used is saved in the results_folder

        Parameters:
        Y (list): labels
        saving_folder (str): path to the folder where the results are saved

        Returns:
        list: list of labels encoded
        """
        Y_encoded = []
        if self.yaml.get('labels_mapping_path'):
            with open(self.yaml.get('labels_mapping_path'), 'r') as file:
                data = json.load(file)
                unique_labels = list(data.values())
                encoded_labels = list(data.keys())
                for el in Y:
                    idx = unique_labels.index(el)
                    Y_encoded.append(int(encoded_labels[idx]))
            shutil.copyfile(self.yaml.get('labels_mapping_path'), os.path.join(saving_folder,"labels_mapping.json"))
        else:
            label_encoder = LabelEncoder()
            Y_encoded = label_encoder.fit_transform(Y)
            self.save_labels_encoding(label_encoder, saving_folder)
        return Y_encoded
    
    
    def get_data(self, X = None, Y = None, split_info = None, data_path = None, split = None):
        """
        Get train and test data.

        Parameters:
        X (list, optional): list of signals read. Defaults to None.
        Y (list, optional): list of labels. Defaults to None.
        split_info (list, optional): list of split info. Defaults to None.
        split (str): 'train'/'test'
        data_path (str, optional): path to dataset. Defaults to None.

        Returns:
        list of tuple, list of tuple: list of tuple of signals and the correspondig labels for train and test
        """
        # if we have the signal already read and the information about the train/test split
        if X and Y and split_info:
            data_index = [i for i, s in enumerate(split_info) if s == split]
            x_data = [X[i] for i in data_index]
            y_data = [Y[i] for i in data_index]

            if len(data_index) < 1:
                logger.error(f"Error. N° data data: {len(x_data)}.")
                exit()
        # if we need to read the audio from the data store
        else:
            x_data, y_data = [], []
            unique_data_labels = os.listdir(os.path.join(data_path, split))
            for label in unique_data_labels:
                audio_files = os.listdir(os.path.join(data_path, split, label))
                for audio in audio_files:
                    signal, sr = sf.read(os.path.join(data_path, split, label,audio))
                    x_data.append(signal)
                    y_data.append(label)
                    
        return [(x_data[i], y_data[i]) for i in range(0, len(x_data))]
                    
        
    def save_labels_encoding(self, label_encoder, saving_folder):
        """
        Saving the labels encoding used
        
        Parameters:
        label_encoder: fitted label encoder.
        saving_folder (str): path to the folder where the results are saved

        Returns:
        None
        """
        label_mapping = {numeric_label: original_label for original_label, numeric_label in
                              zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
        labels_mapping_path = os.path.join(saving_folder, "labels_mapping.json")
        with open(labels_mapping_path, 'w') as file:
            json.dump(label_mapping, file)
        


    def data_preparation(self, data, saving_folder):
        """
        Data preprocessing.
        
        Parameters:
        data (list): lis of tuple [(x1,y1),(x2,y2),...]
        saving_folder (str): path to the folder where the results are saved

        Returns:
        list, list: list of x and y ready to be fed to the classificator
        """
        data_augment = self.yaml.get('data_augment')
        same_class = self.yaml.get('same_class')
        mix_perc = self.yaml.get('mix_perc')
        mix_alpha = self.yaml.get('mix_alpha')
        mix_crop_length = self.yaml.get('mix_crop_length')
        mix_batch_size = self.yaml.get('mix_batch_size')


        X, Y = [], []
        for x,y in tqdm(data, total = len(data), desc = "Features extraction"):
            X.append(self.get_features(x))
            Y.append(y)
        Y = self.get_labels_encoding(Y, saving_folder)

        #data is array of size len(segments)
        #data[0] is array of (1) array of size 32000 sound (2) string label source
        #X is array of len(segments) for which each element array of size 128 vggish features
        
        if data_augment == 'None':
            return X,Y

        elif data_augment == 'MixUp':
            #train data input
            # get two samples of same class mix them
            # do this for percentage of the data
            #check if this data_path refers to train data alone
            #add mixed data to normal data

            X_aug = [sample[0] for sample in data]


            generator = MixUpGenerator(X_aug, Y, batch_size=mix_batch_size, mix_perc= mix_perc, same_class = same_class, 
                                             alpha = mix_alpha, crop_length = mix_crop_length)()

            generator_list = list(generator)
            print(len(generator_list))

            for idx, data in enumerate(generator):  
                print("processing batch {idx}: {data}")
                augmented_X,augmented_Y = data
                for audio_batch, augmented_y in zip(augmented_X, augmented_Y):
                    with self.features_extractor.graph.as_default():
                        [embedding_batch] = self.sess.run([self.features_extractor.embedding_tensor], feed_dict={self.features_extractor.features_tensor: audio_batch})
                        flattened_batch = flatten(embedding_batch)
                        
                        X.append(flattened_batch)
                        Y.append(augmented_y)
            return X,Y


        elif data_augment == 'SpecMix':
            return X,Y



    def compute_scores(self, Y, Y_predicted):
        """
        Computing metrics(accuracy/macro F1 score)

        Parameters:
        Y (array): ground truth
        Y_predicted (array): labels predicted

        Returns:
        float, float: accuracy and f1 score
        """
        accuracy = accuracy_score(y_true=Y, y_pred=Y_predicted)
        f1 = f1_score(y_true=Y, y_pred=Y_predicted, average='macro')
        return accuracy, f1