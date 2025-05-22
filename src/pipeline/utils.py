""" Script for useful functions """
    
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import pandas as pd
import numpy as np
import json
import librosa
import soundfile as sf
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import seaborn as sns
import os
import torch.nn as nn
import torchaudio
import torch
import numpy as np
import logging
from enum import Enum
import threading
from collections import defaultdict
from src.models.vggish_modules import vggish_input
import random
import math
import torch.nn.functional as F






# UNWANTED_LABELS = ["Undefined"]
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

handler = logging.FileHandler("log.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_yaml(yaml_path: str) -> dict:
    """Function used to load the yaml content. Useful for reading configuration files or any other data stored in YAML format.

    Args:
        yaml_path (str): The absolute path to the yaml

    Returns:
        dict: The yaml content
    """
    with open(yaml_path, 'r') as file:
        try:
            yaml_content = yaml.safe_load(file)
            return yaml_content
        except yaml.YAMLError as e:
            logger.error(e)
            return None
        

def create_exp_dir(name: str, model: str, task: str) -> str:
    """
    Function to create a unique experiment directory. Useful for organizing experiment results by model and task.

    Args:
        name (str): The base name for the experiment directory.
        task (str): The name of the task associated with the experiment.
        model (str): The name of the model associated with the experiment.

    Returns:
        str: The path to the created experiment directory.
    """
    parent_path = Path(f'runs/{model}/{task}')
    parent_path.mkdir(exist_ok=True, parents=True)
    exp_path = str(parent_path / name) + "_{:02d}"
    i = 0
    while Path(exp_path.format(i)) in list(parent_path.glob("*")):
        i += 1
    exp_path = Path(exp_path.format(i))
    exp_path.mkdir(exist_ok=True)
    return str(exp_path)


def process_audio_for_inference(path_audio: str, desired_sr: float, desired_duration: float) -> tuple[torch.Tensor, float, float]:
    """Processes audios for inference purposes ensuring each segment is of desired duration.

    Args:
        path_audio (str): Path to the audio that needs to be processed
        desired_sr (float): The desired sampling rate
        desired_duration (float): The desired duration in seconds
        desired_duration (float): The desired duration in seconds

    Raises:
        ValueError: In case the sampling rate of a signal is lower than the desired one.

    Returns:
        tuple: The processed signal tensor, updated sampling rate, and the original audio duration
        tuple: The processed signal tensor, updated sampling rate, and the original audio duration
    """
    y, sr = torchaudio.load(path_audio)


    if sr < desired_sr:
        raise ValueError(f"Sampling rate of {sr} Hz is lower than the desired sampling rate of {desired_sr} Hz.")
    
    if sr != desired_sr:
        y = librosa.resample(y=y.detach().numpy(), orig_sr=sr, target_sr=32_000)
        y = torch.Tensor(y)
        sr = desired_sr 

    length = int(desired_duration * sr)

    total_length = y.size(1)
    num_chunks = (total_length + length - 1) // length  # This rounds up to ensure all data is included

    # Extend y to match the exact multiples of 'length', in order to torch.unfold to generate all the chunks
    if total_length < num_chunks * length:
        padding_size = num_chunks * length - total_length
        y = torch.nn.functional.pad(y, (0, padding_size))

    y = y.unfold(dimension=1, size=length, step=length)

    return y, sr


def save_confusion_matrix(unique_labels, exp_folder, true_labels, predicted_labels, title = "confusion_matrix"):
    """
    Saves the confusion matrix and the normalized confusion matrix as SVG files in the specified folder.

    Parameters:
    unique_labels (list): List of unique labels used in the classification.
    exp_folder (str): Path to the folder where the files will be saved.
    true_labels (list or array): True labels of the data.
    predicted_labels (list or array): Labels predicted by the model.
    title (str, optional): Title for the confusion matrix plots. Default is "confusion_matrix".

    Returns:
    None
    """
    plt.rcParams.update({'font.size': 22})
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{title} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, f'{title}.png'))

    #normalized cm
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{title} - Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, f'{title}-normalized.png'))
    plt.close()


def save_subset_confusion_matrix(all_labels, exp_folder, true_labels, predicted_labels, title = "confusion_matrix"):

    plt.rcParams.update({'font.size': 22})
    cm = confusion_matrix(true_labels, predicted_labels, labels = [0,1,2,3,4,5,6,7,8,9])
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=all_labels, yticklabels=all_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{title} - Confusion Matrix')
    #plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, f'{title}.png'))

    #normalized cm
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=all_labels, yticklabels=all_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{title} - Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, f'{title}-normalized.png'))
    plt.close()



def save_training_curves(exp_folder, train_losses, val_losses, val_accs):
    """
    Saves the training and validation curves as a PNG file in the specified folder.

    Parameters:
    exp_folder (str): Path to the folder where the file will be saved.
    train_losses (list or array): Training losses per epoch.
    val_losses (list or array): Validation losses per epoch.
    val_accs (list or array): Validation accuracies per epoch.

    Returns:
    None
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(exp_folder, f'training_curves.png'))


def flatten(array):
    """
    Flatten a NumPy array.

    Parameters:
    - array (numpy.ndarray): The input array to be flattened.

    Returns:
    - flatten_array (numpy.ndarray): The flattened array.
    """
    flatten_array = array.flatten()
    return flatten_array


def process_data_for_inference(path_audio: str, desired_sr: float, desired_duration: float):
    """
    Process a single signal by resampling and segmenting or padding it.


    Parameters:
    signal (np.array): Signal array.
    original_sr (float): Original sampling rate of the signal.

    Returns:
    list: List of processed segments.
    """
    signal, original_sr = sf.read(path_audio)
    # Resample the signal if the original sampling rate is different from the target
    if original_sr != desired_sr:
        signal = librosa.resample(y=signal, orig_sr=original_sr, target_sr=desired_sr)
        
    # Pad the signal if it is shorter than the segment length   
    segment_length = math.ceil(desired_duration*desired_sr)
    if len(signal) < segment_length:
        delta = segment_length - len(signal)
        delta_start = delta // 2
        delta_end = delta_start if delta%2 == 0 else (delta // 2) + 1 
        segments = np.pad(signal, (delta_start, delta_end), 'constant', constant_values=(0, 0))
        # Segment the signal if it is longer or equal to the segment length
    elif len(signal) >= segment_length:
        segments = []
        # Calculate the number of full segments in the signal
        n_segments = len(signal)//(segment_length)
        # Extract each segment and append to the list
        for i in range(n_segments):
            segment = signal[(i*segment_length):((i+1)*segment_length)]
            segments.append(segment)
        
    return segments

class AugmentMelSTFT(nn.Module):
    """ This class is used in order to generate the mel spectrograms for the EffAT and PaSST models.
        It includes additional features to the normal mel such as maskings and augmentation.
    """
    def __init__(self, n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48, timem=192,
                 fmin=0.0, fmax=None, fmin_aug_range=10, fmax_aug_range=2000):
        """The constructor for AugmentMelSTFT class.

        Args:
            n_mels (int, optional): The number of mel frequency bands. Defaults to 128.
            sr (int, optional): The sampling rate. Defaults to 32000.
            win_length (int, optional): The length of the window. Defaults to 800.
            hopsize (int, optional): Hop size. Defaults to 320.
            n_fft (int, optional): The number of FFT points. Defaults to 1024.
            freqm (int, optional): The frequency masking parameter. Defaults to 48.
            timem (int, optional): The time masking parameter. Defaults to 192.
            fmin (float, optional): The min frequency. Defaults to 0.0.
            fmax (float, optional): The max frequency to be plot. Defaults to None (if None, its computed such as sr / 2).
            fmin_aug_range (int, optional): The min augmentation range. Defaults to 10.
            fmax_aug_range (int, optional): The max augmentation range. Defaults to 2000.
        """
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
            logger.warning(f"Warning: FMAX is None setting to {fmax} ")
        self.fmax = fmax
        self.hopsize = hopsize
        self.register_buffer('window',
                             torch.hann_window(win_length, periodic=False),
                             persistent=False)
        assert fmin_aug_range >= 1, f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert fmax_aug_range >= 1, f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)


    def forward(self, x):
        # x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)  # Makes the mels look bad
        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=False)
        x = (x ** 2).sum(dim=-1)  # power mag
        # GOOD ONES
        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()
        
        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels,  self.n_fft, self.sr,
                                        fmin, fmax, vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()

        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.  # fast normalization

        return melspec


class EffATWrapper(nn.Module):
    """This class is used as a wrapper to the EffAT model found in src/models/effat_repo. The wrapper
    replaces the last layer of the model and changes it for a new one with the specified number of classes and if desired it frozes all layers
    except the last one.
    """
    def __init__(self, num_classes: int, model, freeze: bool):
        """The constructor for the EffATWrapper class.

        Args:
            num_classes (int): The number of classes for the model.
            model (_type_): The EfficientAT architecture loaded with the get_mn(pretrained_name=self.name_model) or get_dymn(pretrained_name=self.name_model) function
            freeze (bool): A boolean to check if we want to freeze all layers except the last one or not.
        """
        super(EffATWrapper, self).__init__()
        self.num_classes = num_classes
        self.model = model
        self.gradients = None
        self.activations = None


        #for grad CM visualization hook the last conv layer of features
        final_conv = self.model.features[-1]
        final_conv.register_forward_hook(self._save_activation)
        final_conv.register_full_backward_hook(self._save_gradient)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False 
        
        # Replace the number of output features to match classes
        new_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=960, out_features=1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=self.num_classes, bias=True)
        )
        model.classifier = new_classifier
        self.model = model
        

    def forward(self, melspec):
        logits = self.model(melspec)
        return logits
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

class ConfusionMatrix:
    def __init__(self, labels_mapping_path: str) -> None:
        """
        Initialize the ConfusionMatrix class.
        """
        self.labels_mapping_path = labels_mapping_path

    def plot(self, y_true, y_pred):
        """
        Plot the confusion matrix.

        Parameters:
        - y (numpy.ndarray): True labels.
        - y_pred (numpy.ndarray): Predicted labels.

        Returns:
        - fig (matplotlib.figure.Figure): The generated matplotlib figure.
        """
        # Compute confusion matrix and normalize
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        
   
        with open(self.labels_mapping_path, 'r') as file:
            label_mapping = json.load(file)
            y_true = np.array([label_mapping[str(label)] for label in y_true])
            y_pred = np.array([label_mapping[str(label)] for label in y_pred])
        
        if len(list(np.unique(y_true)))>len(list(np.unique(y_pred))):
            labels = np.unique(y_true)
        else:
            labels = np.unique(y_pred)
        
        cm_percent = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 9))

        # Plot the normalized confusion matrix
        cm_display_percent = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=labels)
        cm_display_percent.plot(ax=ax1, cmap='Blues', values_format='.2%')

        # Plot the absolute confusion matrix
        cm_display_absolute = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
        cm_display_absolute.plot(ax=ax2, cmap='Blues', values_format='d')
        
        # Rotate labels in the second plot
        ax1.set_xticklabels(labels, rotation=45)
        ax1.set_yticklabels(labels, rotation=45)
        ax2.set_xticklabels(labels, rotation=45)
        ax2.set_yticklabels(labels, rotation=45)
        # Set a title for the entire figure
        
        fig.suptitle('Confusion Matrix', fontsize=11)

        return fig
    
    def save_plot(self, plot, saving_folder):
        plot.savefig(os.path.join(saving_folder, "ConfusionMatrix.png"))
        

class ValidationPlot:
    def __init__(self, gridsearch):
        self.gridsearch = gridsearch
        self.grid_results = gridsearch.cv_results_
        self.best_parameters = gridsearch.best_params_
        self.grid_params = self.grid_results['param_C']
        
    def get_filtered_dataset(self, df, filters):
        idx_to_drop = []
        for key, value in filters.items():
            for idx, row in df.iterrows():
                grid_key = 'param_' + key
                if row[grid_key] != value:
                    idx_to_drop.append(idx)
        unique_idx_to_drop = list(set(idx_to_drop))
        df_filtered = df.drop(unique_idx_to_drop)
        return df_filtered

    def plot(self):
        metric = 'accuracy'
        best_params_show = self.best_parameters['C']

        del self.best_parameters['C']
        df_grid = pd.DataFrame(self.grid_results)

        if len(self.best_parameters) != 0:
            df_grid = self.get_filtered_dataset(df=df_grid, filters=self.best_parameters)

        fig, ax = plt.subplots(figsize=(8, 6))

        validation_metric = df_grid['mean_test_score'].to_numpy()
        training_metric = df_grid['mean_train_score'].to_numpy()
 
        param_values = np.unique(np.array(list(self.grid_results['param_C'])))
        
        lim0_val = df_grid['mean_test_score'].to_numpy()-df_grid['std_test_score'].to_numpy()
        lim1_val = df_grid['mean_test_score'].to_numpy()+df_grid['std_test_score'].to_numpy()
    
        ax.fill_between(param_values, lim0_val, lim1_val, alpha=0.4, color = 'skyblue')
        ax.plot(param_values, validation_metric, marker='o', color='b', label='Validation ' + metric)
        
        lim0_tr = df_grid['mean_train_score'].to_numpy()-df_grid['std_train_score'].to_numpy()
        lim1_tr = df_grid['mean_train_score'].to_numpy()+df_grid['std_train_score'].to_numpy()
        
        ax.fill_between(param_values, lim0_tr, lim1_tr, alpha=0.4, color = 'lightcoral')
        ax.plot(param_values, training_metric, marker='o', color='r', label='Training ' + metric)

        ax.axvline(best_params_show, color ='green', linestyle= '--', alpha = 0.75, label = 'Best C') 
        
        ax.set_xlabel('C', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        fig.suptitle(f'Training and Validation {metric} for VGGish-SVM', fontsize=14, fontweight='bold' )
        ax.legend(loc = 'best', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig

    def save_plot(self, plot, saving_folder):
        plot.savefig(os.path.join(saving_folder, "TrainigCurves.png"))


def visualize_inference(path_json: str, path_audio: str, path_yaml: str, model: str) -> None:
    yaml_content = load_yaml(path_yaml)

    if os.path.isfile(path_json) and os.path.isfile(path_json):
        with open(path_json, 'r') as f:
            results = json.load(f)

        y, sr = librosa.load(path_audio, sr=None)
        
        #add necessary yaml content to vggish inference
        if 'sr' not in yaml_content:
            if 'desired_sr' in yaml_content:
                yaml_content['sr'] = yaml_content['desired_sr']
            else:
                raise Exception("No sampling rate or desired sampling rate found in the yaml file.")  
        if sr >= yaml_content["sr"]:
            y = librosa.resample(y=y, orig_sr=sr, target_sr=yaml_content["sr"])
        else:
            raise Exception(f"Sampling rate is lower than {yaml_content['sr']} Hz")

        S = librosa.feature.melspectrogram(y=y, sr=yaml_content["sr"], n_mels=128, hop_length=yaml_content["hopsize"], n_fft=yaml_content["n_fft"])
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        img = librosa.display.specshow(S_dB, sr=yaml_content["sr"], hop_length=yaml_content["hopsize"], x_axis='time', y_axis='linear', ax=ax)
        plt.title(os.path.basename(path_audio))
        
        max_time = y.shape[0] / yaml_content["sr"]

        if 'duration' not in yaml_content:
            if 'desired_duration' in yaml_content:
                yaml_content['duration'] = yaml_content['desired_duration']
            else:
                raise Exception("No duration or desired duration found in the yaml file.")
            
        line_positions = np.arange(0, max_time-1, yaml_content["duration"])
        
        if model == 'effat':
            predicted_classes = [value["Predicted Class"] for key, value in results.items()]
        elif model == 'passt':
            predicted_classes = [entry["Predicted Class"] for entry in results]
        elif model == 'vggish':
            predicted_classes = list(results['class by time interval'].values())


        for i, pos in enumerate(line_positions):
            ax.axvline(x=pos, color='red', linewidth=1)
            ax.text(pos + yaml_content["duration"] / 2, S.shape[0] * 10, predicted_classes[i], color='white', verticalalignment='top', rotation=90)

        plt.set_cmap('gray')
        plt.show()

    elif os.path.isdir(path_json) and os.path.isdir(path_audio):
        audios = [os.path.join(path_audio, audio) for audio in os.listdir(path_audio)]
        for audio in audios:
            y, sr = librosa.load(audio, sr=None)
            if sr >= yaml_content["sr"]:
                y = librosa.resample(y=y, orig_sr=sr, target_sr=yaml_content["sr"])
            else:
                raise Exception(f"Sampling rate is lower than {yaml_content['sr']} Hz")

            S = librosa.feature.melspectrogram(y=y, sr=yaml_content["sr"], n_mels=128, hop_length=yaml_content["hopsize"], n_fft=yaml_content["n_fft"])
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            img = librosa.display.specshow(S_dB, sr=yaml_content["sr"], hop_length=yaml_content["hopsize"], x_axis='time', y_axis='mel', ax=ax)
            plt.title(os.path.basename(audio))
            
            max_time = y.shape[0] / yaml_content["sr"]
            line_positions = np.arange(0, max_time-1, yaml_content["duration"])
            
            path_jsonfile = os.path.join(path_json, f'predictions_{os.path.splitext(os.path.basename(audio))[0]}.json')
            with open(path_jsonfile, 'r') as f:
                results = json.load(f)
            
            if model == 'effat':
                predicted_classes = [value["Predicted Class"] for key, value in results.items()]
            elif model == 'passt':
                predicted_classes = [entry["Predicted Class"] for entry in results]
            elif model == 'vggish':
                predicted_classes = list(results['class by time interval'].values())


            for i, pos in enumerate(line_positions):
                ax.axvline(x=pos, color='red', linewidth=1)
                ax.text(pos + yaml_content["duration"] / 2, S.shape[0] * 10, predicted_classes[i], color='white', verticalalignment='top', rotation=90)

            plt.show()
        
    else:
        logger.error("path_json and path_audio should be both file or folder")


def generate_gradcam_map(activations, gradients):
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # Global average pooling
    gradcam_map = (weights * activations).sum(dim=1, keepdim=True)
    gradcam_map = F.relu(gradcam_map)

    # Normalize and resize
    gradcam_map -= gradcam_map.min()
    gradcam_map /= gradcam_map.max() + 1e-8
    gradcam_map = gradcam_map.squeeze().cpu().numpy()
    return gradcam_map


def visualize_gradcam(cam: torch.Tensor, mel_input: torch.Tensor, save_path: str, true_label: int=None, pred_label: int = None):
    """
    Overlays Grad-CAM on top of the input mel spectrogram and saves the visualization.
    Args:
        cam (Tensor): Grad-CAM map, shape (1, H, W)
        mel_input (Tensor): Mel spectrogram, shape (1, H, W)
        save_path (str): Path to save the output image
        true_label (int, optional): True label for the audio. Defaults to None.
        pred_label (int, optional): Predicted label for the audio. Defaults to None.

    """

    if isinstance(cam, torch.Tensor):   
        cam = cam.squeeze().detach().cpu().numpy()
    else:       
        cam=cam.squeeze()

    mel = mel_input.squeeze().detach().cpu().numpy()

    # Resize CAM to match mel shape using np.interp (if shapes differ)
    if cam.shape != mel.shape:
        cam_resized = np.zeros_like(mel)
        for i in range(cam.shape[0]):
            cam_resized[i, :] = np.interp(
                np.linspace(0, cam.shape[1] - 1, mel.shape[1]),
                np.arange(cam.shape[1]),
                cam[i]
            )
        cam = cam_resized

    
    # Calculate time and frequency ranges for the mel spectrogram
    sr = 16000  # Sampling rate
    hop_length = 160  # Hop length
    duration = mel.shape[1] * hop_length / sr  # Total duration in seconds
    max_freq = sr / 2  # Nyquist frequency
    extent = [0, duration, 0, max_freq ]

    # Save the original mel spectrogram
    original_path = os.path.join(save_path, "original.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax, fmax=max_freq)
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.title(f'Original Mel Spectrogram with label{true_label} and prediction {pred_label}')
    plt.tight_layout()
    plt.savefig(original_path)  # Save original mel spectrogram
    plt.close()
 

    #save the gradcam only
    gradcam_only_path = os.path.join(save_path, "gradcam_only.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax, fmax=max_freq)
    ax.imshow(cam, cmap='jet', alpha=0.4, aspect='auto', origin='lower', extent=extent)
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.title(f'Mel Spectrogram with Grad-CAM Overlay with label{true_label} and predicted label{pred_label}')
    plt.tight_layout()
    plt.savefig(gradcam_only_path)  # Save Grad-CAM overlay
    plt.close()


    # Save the mel spectrogram with Grad-CAM overlay
    gradcam_path = os.path.join(save_path, "gradcam.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax, fmax=max_freq)
    ax.imshow(cam, cmap='jet', alpha=0.4, aspect='auto', origin='lower', extent=extent)
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.title(f'Mel Spectrogram with Grad-CAM Overlay with label{true_label} and predicted label{pred_label}')
    plt.tight_layout()
    plt.savefig(gradcam_path)  # Save Grad-CAM overlay
    plt.close()
    

class SuperpositionType(Enum):
    """A helper class to check the type of superposition when dealing with the overlapping.
    """
    NO_SUPERPOSITION = 0
    STARTS_BEFORE_AND_OVERLAPS = 1
    STARTS_AFTER_AND_OVERLAPS = 2
    CONTAINS = 3
    IS_CONTAINED = 4


class LibrosaSpec(nn.Module):
    def __init__(self, mel: bool, sr: float = 32_000, win_length: int = 800, hopsize: int = 320, n_fft: int = 1024, n_mels: int = 128):
        torch.nn.Module.__init__(self)  # To make the class callable
        self.mel = mel
        self.sr = sr
        self.win_length=win_length
        self.hopsize=hopsize
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, y):
        y = y.cpu().detach().numpy()
        if self.mel:
            S = librosa.feature.melspectrogram(y=y,
                                               sr=self.sr,
                                               n_mels=self.n_mels,
                                               hop_length=self.hopsize,
                                               win_length=self.win_length,
                                               n_fft=self.n_fft)
            S_dB = librosa.power_to_db(S)

        else:
            S = np.abs(librosa.stft(y,
                                    n_fft=self.n_fft,
                                    hop_length=self.hopsize,
                                    win_length=self.win_length))
            S_dB = librosa.amplitude_to_db(S, ref=np.max)

        S_normalized = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())  # Data between 0 and 1
        return torch.Tensor(S_normalized).to(self.device)


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    # python 3
    def __next__(self):
        with self.lock:
            return self.it.__next__()

    # python 2
    #def next(self):
    #    with self.lock:
    #        return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

def masking(spec, ratio=0.3, t_times= np.random.randint(3), f_times= np.random.randint(3)) :
    mask = np.ones(spec.shape)

    for _ in range(t_times) :
        t = np.random.randint(0, (1-ratio)*mask.shape[0]+1)
        mask[:, t:t+int(mask.shape[0]*ratio)] = 0

    for _ in range(f_times) :
        f = np.random.randint(0, (1-ratio)*mask.shape[1]+1)
        mask[f:f+int(mask.shape[1]*ratio), :] = 0
    inv_mask = -1 * (mask - 1)

    return mask, inv_mask



class MixUpGenerator:
    ''' Reference: https://github.com/yu4u/mixup-generator
    '''	

    def __init__(self, X_aug, Y_aug, sample_rate = 16000, shuffle=True, mix_perc=0.1, same_class=True, 
                 alpha = 0.1, batch_size=32, datagen=None, crop_length=400):
        self.X = X_aug
        self.Y = Y_aug
        self.sample_rate = sample_rate
        self.mix_perc = mix_perc
        self.same_class = same_class
        self.sample_num = len(X_aug)
        self.lock = threading.Lock()
        self.shuffle = shuffle
        self.alpha = alpha
        self.batch_size = batch_size
        self.datagen = datagen
        self.NewLength = crop_length
        self.swap_inds = [1, 0, 3, 2, 5, 4]
        
 
    def __iter__(self):
        return self

    @threadsafe_generator
    def __call__(self):
        with self.lock:
            while True:
                indexes = self.__get_exploration_order()
                itr_num = int(len(indexes) // (self.batch_size * 2))

                for i in range(itr_num):
                    batch_ids = indexes[int(i * self.batch_size * 2):int((i + 1) * self.batch_size * 2)]
                    # X, y = self.__data_generation(batch_ids)

                    # yield X, y
                    yield self.__data_generation(batch_ids)
    
    
    def __get_exploration_order(self):
        #If same class is True only make pairs of samples with same class, otherwise shuffle all indeces
        if not self.same_class:
            indexes = np.arange(self.sample_num)
            if self.shuffle:
                np.random.shuffle(indexes)
            return indexes
        
        else:
            label_to_indices = defaultdict(list)
            for i, label in enumerate(self.Y):
                label_to_indices[label].append(i)

            target_augmented_samples = int(self.mix_perc * self.sample_num)

            # Create pairs of indices with the same label, label already encoded in input Y
            #TODO aanpassen totdat percentage is bereikt van hoeveel neppe samples er moeten komen, anders vel te lang
            paired_indices = []
            for indices in label_to_indices.values():
                if len(indices) >= 2:
                    num_pairs_needed = target_augmented_samples - len(paired_indices)
                    if num_pairs_needed <=0:
                        break
                    possible_pairs = [(indices[i], indices[j]) for i in range(len(indices)) for j in range(i + 1, len(indices))]
                    num_pairs_to_sample = min(len(possible_pairs), num_pairs_needed)
                    paired_indices.extend(random.sample(possible_pairs, num_pairs_to_sample))

        # Shuffle the pairs
        np.random.shuffle(paired_indices)
        # Flatten the list of pairs into a single list of indices
        flat_indices = [index for pair in paired_indices for index in pair]

        return np.array(flat_indices)
    
    def __data_generation(self, batch_ids):
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape((self.batch_size, 1, 1, 1))
        y_l = l.reshape((self.batch_size, 1))


        X1 = [self.X[i] for i in batch_ids[:self.batch_size]]
        X2 = [self.X[i] for i in batch_ids[self.batch_size:]]



        #STILL ERRORING HERE
        #convert signal array to melspec
        for i in range(len(X1)):
            X1[i] = vggish_input.waveform_to_examples(X1[i], self.sample_rate)
            X2[i] = vggish_input.waveform_to_examples(X2[i], self.sample_rate)
    

        if self.NewLength > 0:
            for j in range(X1.shape[0]):
                StartLoc1 = np.random.randint(0, X1.shape[2] - self.NewLength)
                StartLoc2 = np.random.randint(0, X2.shape[2] - self.NewLength)

                X1[j, :, 0:self.NewLength, :] = X1[j, :, StartLoc1:StartLoc1 + self.NewLength, :]
                X2[j, :, 0:self.NewLength, :] = X2[j, :, StartLoc2:StartLoc2 + self.NewLength, :]

                if X1.shape[-1] == 6:
                    # randomly swap left and right channels
                    if np.random.randint(2) == 1:
                        X1[j, :, :, :] = X1[j:j + 1, :, :, self.swap_inds]
                    if np.random.randint(2) == 1:
                        X2[j, :, :, :] = X2[j:j + 1, :, :, self.swap_inds]

            X1 = X1[:, :, 0:self.NewLength, :]
            X2 = X2[:, :, 0:self.NewLength, :]

        X = X1 * X_l + X2 * (1.0 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1.0 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1.0 - y_l)

        return X, y
  
def chunk_vizualisation(true_labels, predicted_labels, data_loader, nr_classes, mel_transform, output_folder, num_visualizations=1):
    """
    Visualize mel spectrograms for each class where true_label == predicted_label and true_label != predicted_label.

    Args:
        true_labels (list): List of true labels.
        predicted_labels (list): List of predicted labels.
        data_loader (DataLoader): DataLoader containing the audio inputs.
        nr_classes (int): Number of classes.
        mel_transform (AugmentMelSTFT): Mel spectrogram transformation.
        output_folder (str): Path to the folder where the spectrograms will be saved.
        num_visualizations (int): Number of matches/mismatches to visualize per class.

    Returns:
        None
    """
 
    os.makedirs(output_folder, exist_ok=True)

    # Precompute indices for each class
    class_indices = defaultdict(lambda: {"correct": [], "incorrect": []})
    for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)):
        if true == pred:
            class_indices[true]["correct"].append(i)
        else:
            class_indices[true]["incorrect"].append(i)

    # Iterate over all classes
    for cls in range(nr_classes):
        # Get correct and incorrect indices for the current class
        correct_indices = class_indices[cls]["correct"]
        incorrect_indices = class_indices[cls]["incorrect"]

        # Randomly select up to `num_visualizations` samples for correct predictions
        if correct_indices:
            selected_correct_indices = random.sample(correct_indices, min(len(correct_indices), num_visualizations))
            for idx in selected_correct_indices:
                audio_input, _, _ = data_loader.dataset[idx]
                mel_spec = mel_transform(audio_input.unsqueeze(0))  # Generate mel spectrogram using AugmentMelSTFT
                plt.figure(figsize=(10, 4))
                plt.imshow(mel_spec.squeeze().cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
                plt.title(f"Class {cls} - Correct Prediction")
                plt.colorbar()
                plt.savefig(os.path.join(output_folder, f"class_{cls}_correct_{idx}.png"))
                plt.close()

        # Randomly select up to `num_visualizations` samples for incorrect predictions
        if incorrect_indices:
            selected_incorrect_indices = random.sample(incorrect_indices, min(len(incorrect_indices), num_visualizations))
            for idx in selected_incorrect_indices:
                audio_input, _, _ = data_loader.dataset[idx]
                wrong_target = predicted_labels[idx]
                mel_spec = mel_transform(audio_input.unsqueeze(0))  # Generate mel spectrogram using AugmentMelSTFT
                plt.figure(figsize=(10, 4))
                plt.imshow(mel_spec.squeeze().cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
                plt.title(f"Class {cls} - Incorrect Prediction - confused with {wrong_target}")
                plt.colorbar()
                plt.savefig(os.path.join(output_folder, f"class_{cls}_incorrect_confused_with_{wrong_target}.png"))
                plt.close()
