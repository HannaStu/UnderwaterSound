"""
Module for the File class, for Sound Signature datasets

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

# Buildin modules
import datetime as dt
import logging
import os

import antropy as ant
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# External modules
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

# Internal modules
from soundsignature.annotation import Annotation
from soundsignature.extract_meta import read_metadata


class File:

    def __init__(
        self,
        file_path: str,
        frequency: int = -1,
        compressed: str = "No",
        encoding: str = "None",
        byte_count: int = -1,
        byte_rate: int = -1,
        byte_sample: int = -1,
        start_time: dt.datetime = dt.datetime(1970, 1, 1),
        end_time: dt.datetime = dt.datetime(1970, 1, 1),
        format: str = "None",
        sample_width: float = np.NaN,
        file_duration: float = np.NaN,
        annotations: list[Annotation] = [],
    ) -> None:
        """This class contains information about a recorded audio data file. A single file can contain several samples.

        Parameters
        ----------
        file_path : str
            Path of current file. Relative to the global defined DATADIR.
        frequency : int
            Sampling frequency of the audio file, in Hertz [Hz].
        compressed : str
            Whether the file has been compressed or not (None, Lossless, Compressed).
        encoding : str
            Type of encoding of the file.
        byte_count : int
            Number of Byte of the sample.
        byte_rate : int
            In [bit/s].
        byte_sample : int
            Quantification level (16 or 24 bits depending on the selected format).
        start_time : dt.datetime
            Audio start timestamp. UTC DateTime in ISO 8601 format: YYYY-MM-DDThh:mm[:ss]
        end_time : dt.datetime
            Audio end timestamp. UTC DateTime in ISO 8601 format: YYYY-MM-DDThh:mm[:ss]
        format : str
            e.g. WAV, FLAC etc.
        sample_width : float
            ???
        file_duration : float
            Length of audio file in seconds [s].
        annotations : list[Annotation]
            List of soundsignature.Annotations associated to this soundsignature.File.
        """
        self.file_path = file_path
        self.frequency = frequency
        self.compressed = compressed
        self.sample_width = sample_width
        self.encoding = encoding
        self.byte_count = byte_count
        self.byte_rate = byte_rate
        self.byte_sample = byte_sample
        self.start_time = start_time
        self.end_time = end_time
        self.format = format
        self.file_duration = file_duration
        self.annotations = annotations
        self.audio_array = None
        self.snr = None
        self.entropy = None

    def __repr__(self):
        file_name = os.path.basename(self.file_path)
        return f"File [{file_name}]"

    def read_wav_file(self, start_datetime=None):
        """reading wav file of current instance, returning a pd.DataFrame"""

        # If file is already read, return it directly, to improve performance
        if self.audio_array is not None:
            return self.audio_array

        if start_datetime is None:
            start_datetime = self.start_time

        # The filepath is always relative to the globally set DATADIR
        fullfilepath = os.path.join(os.getenv("DATADIR"), self.file_path)

        self.audio_array = read_wav(fullfilepath, start_datetime)
        return self.audio_array

    def make_plot(
        self,
        start_interval=None,
        end_interval=None,
        list_of_snippets=None,
        start_date=None,
    ):
        """Read wav data and plot it over time"""

        if start_interval is None:
            start_interval = self.start_time
        if end_interval is None:
            end_interval = self.end_time

        # TODO add label information to the plot
        if list_of_snippets is None:
            list_of_snippets = []
        if start_date is None:
            start_date = self.start_time

        if start_interval < start_date:
            logging.error("Start interval is after the start date of the file")
            logging.info("Setting start interval to start date")
            start_interval = start_date
        elif end_interval > start_date + dt.timedelta(seconds=self.file_duration):
            logging.error("End interval is after the end date of the file")
            logging.info("Setting end interval to end date of the file")
            end_interval = start_date + dt.timedelta(seconds=self.file_duration)

        # load data from wav-file
        dataframe_wav_file = self.read_wav_file(start_datetime=start_date)
        # remove values before start_interval
        before_start_interval_booleans = dataframe_wav_file["date"] > start_interval
        # remove values after end_interval
        after_end_interval_booleans = dataframe_wav_file["date"] < end_interval

        # get date and value arrays from interval
        time_array = dataframe_wav_file[
            before_start_interval_booleans & after_end_interval_booleans
        ]["date"]
        voltage = dataframe_wav_file[before_start_interval_booleans & after_end_interval_booleans][
            "value"
        ]

        # make a figure
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time_array, voltage)

        # plot the start of every snippet recording as a vertical line
        for snip in list_of_snippets:
            plt.axvline(x=snip, color="black", linestyle=":")

        # figure settings
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.set_ylabel("Voltage (V)")
        ax.set_xlabel("Time [h:min:s]")
        ax.set_title(
            f"Raw data in the Time domain\nStart recording at {time_array.iloc[0]}\nEnd recording at {time_array.iloc[-1]}"
        )
        ax.grid()

        ax.legend(["Continuous signal", "Start sample"], loc="upper right")

        fig.tight_layout()

        return fig, ax

    def fetch_audio_from_label(self, label_info):
        """Function to obtain the single sample corresponding to the label"""

        # load data from wav-file
        dataframe_wav_file = self.read_wav_file(start_datetime=self.start_time)

        # Specify the time interval of the label
        start_interval = self.start_time + dt.timedelta(seconds=label_info.tmin)
        end_interval = self.start_time + dt.timedelta(seconds=label_info.tmax)

        # remove values before start_interval
        before_start_interval_booleans = dataframe_wav_file["date"] > start_interval
        # remove values after end_interval
        after_end_interval_booleans = dataframe_wav_file["date"] < end_interval

        # Obtain the voltage as function of time for the label
        time_array = dataframe_wav_file[
            before_start_interval_booleans & after_end_interval_booleans
        ]["date"]
        voltage = dataframe_wav_file[before_start_interval_booleans & after_end_interval_booleans][
            "value"
        ]

        return voltage, time_array

    def make_spectrogram(self, segment_time=1, overlap=0.1):
        """
        Read wav data and make spectrogram

        Parameters
        ----------
        segment_time : float
            Time of the segment in seconds.
        overlap : float
            Overlap between segments in fraction of segment_time.

        """

        # load data from wav-file
        dataframe_wav_file = self.read_wav_file()
        # remove values before start_interval
        before_start_interval_booleans = dataframe_wav_file["date"] > self.start_time
        # remove values after end_interval
        after_end_interval_booleans = dataframe_wav_file["date"] < self.end_time

        # get date and value arrays from interval
        voltage = dataframe_wav_file[before_start_interval_booleans & after_end_interval_booleans][
            "value"
        ]

        # spectrogram properties
        segment = segment_time * self.frequency  # window size (in samples)
        noverlap = segment * overlap  # fraction of overlap between segments

        # calculate spectrogram (power spectrum in units V**2)
        f, t, Sxx = signal.spectrogram(
            voltage,
            fs=self.frequency,
            nperseg=segment,
            noverlap=noverlap,
            scaling="spectrum",
        )
        # S_dB = 10 * np.log10(Sxx)

        # time array in datetime
        time_start = self.start_time
        time = time_start + dt.timedelta(seconds=1) * t

        # make a figure
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 4), dpi=100)
        mesh = plt.pcolormesh(time, f, Sxx, cmap="RdBu_r")

        # axes and figure settings
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_tick_params(rotation=45)
        ax.set_ylabel("Frequency [Hz]")
        ax.set_yscale("symlog")
        ax.set_xlabel("Time [h:min:s]")
        ax.set_title(
            f"Spectrogram of [{os.path.basename(self.file_path)}]\n"
            f"Start recording at {time_start.strftime(format='%Y-%m-%d %H:%M:%S')}"
        )

        # colorbar
        cbar = plt.colorbar(mesh, orientation="vertical", location="right")
        cbar.ax.set_ylabel(ylabel="Voltage")

        fig.tight_layout()

        return fig, ax

    def calculate_file_snr_entropy(self, noise_floor, sensitivity, gain):
        """Function for snr and entropy"""
        # load the sound amplitudes of the sample
        logging.info(f"reading wav for file {self.file_path}")
        audio_values, _ = sf.read(self.file_path)

        # # remove values before start_interval
        # before_start_interval_booleans = sample_audio["date"] > self.start_time
        # # remove values after end_interval
        # after_end_interval_booleans = sample_audio["date"] < self.end_time

        # get date and value arrays from interval
        # audio_values = sample_audio[
        #     before_start_interval_booleans & after_end_interval_booleans
        # ]["value"]

        # remove the first few seconds (to remove calibration tones)
        removal_seconds = 3
        no_indices = removal_seconds * self.frequency
        audio_values = audio_values[no_indices:]

        # convert to pressure
        calibration = 10 ** ((sensitivity + gain) / 20) * 1e6  # in V/Pa
        audio_values = audio_values / calibration

        # noise level = noise floor hydrophone
        noise = 10 ** (noise_floor / 20) * 1e-6  # in Pa

        # signal level = 99% percentile of the sample
        signal_99 = np.percentile(abs(audio_values), 99)

        # signal_level = rms of the sample
        signal_rms = np.sqrt(np.mean(audio_values**2))

        # calculate snr
        # add this property to the class structure
        self.snr_99 = 20 * np.log10(signal_99 / noise)
        self.snr_rms = 20 * np.log10(signal_rms / noise)

        ## ENTROPY PART
        # calculate the normalized spectral entropy of the complete sample
        # add this property to the class structure
        self.entropy = ant.spectral_entropy(
            audio_values, self.frequency, method="fft", normalize=True
        )

        return self.snr_99, self.snr_rms, self.entropy

    @classmethod
    def from_frame(cls, fileID, files_dataframe):
        """to return a single file instance associated to fileID."""

        # Get particular, single row, according to fileID
        file_row = files_dataframe.loc[files_dataframe["FileID"] == fileID].squeeze()

        # To verify that station row is just a single row / pd.Series
        if not isinstance(file_row, pd.Series):
            ValueError(f"Multiple rows selected by station ID [{fileID}]")

        # Parse all information of dataframe towards initialization of file object
        return cls.from_series(file_row)

    @classmethod
    def from_series(cls, file_row):
        # Parse all information of dataframe towards initialization of file object
        return cls(
            file_path=file_row["Path"],
            frequency=file_row["frequency"],
            compressed=file_row["compressed"],
            encoding=file_row["encoding"],
            byte_count=file_row["byte count"],
            byte_rate=file_row["byte rate"],
            byte_sample=file_row["byte sample"],
            format=file_row["format"],
            annotations=[],
        )

    @classmethod
    def from_series_only_path(cls, file_row):
        """Function to create a file object from a path"""
        # Parse only information about the path, towards the constructor of the file object
        return cls(file_path=file_row["Path"])

    def to_dict(self) -> dict:
        """Function to return dictionary of the file"""

        return {
            "Path": self.file_path,
            "frequency": self.frequency,
            "compressed": self.compressed,
            "sample_width": self.sample_width,
            "encoding": self.encoding,
            "byte_count": self.byte_count,
            "byte_rate": self.byte_rate,
            "byte_sample": self.byte_sample,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "annotations": self.annotations,
            "duration": self.file_duration,
            "format": self.format,
            "snr": self.snr,
            "entropy": self.entropy,
        }

    def downsample_by_integer(self, factor, export=False):
        """
        Function to downsample the file by an integer factor.

        Args:
            factor (int): The downsampling factor.
            export (bool, optional): Whether to export the downsampled file. Defaults to False.

        Returns:
            data_downsampled: The downsampled data.
            new_samplerate: The new samplerate.
        """
        data, samplerate = sf.read(self.file_path)

        # resample data by keeping every factor-th sample
        data_downsampled = data[::factor]
        # update samplerate
        new_samplerate = int(samplerate / factor)

        logging.info(f"Original samplerate: {samplerate} Hz")
        logging.info(f"New samplerate: {samplerate/factor} Hz")

        # export and update file
        if export:
            new_file_path = self.file_path.replace(".flac", f"_downsampled_by_{factor}.flac")
            sf.write(
                new_file_path,
                data_downsampled,
                new_samplerate,
            )

            self.file_path = new_file_path
            self.frequency = int(samplerate / factor)

            with sf.SoundFile(self.file_path) as file:
                logging.info(
                    f"Converting byte count ({self.byte_count}) and byte rate ({self.byte_rate})"
                )

                # set new byte count and byte rate
                log = file.extra_info.split()
                self.byte_count = int(log[log.index("Length") + 2])
                self.byte_rate = self.byte_count / self.file_duration * 8

                logging.info(
                    f"New byte count = {self.byte_count}, new byte rate = {self.byte_rate}"
                )

        return data_downsampled, new_samplerate

    def downsample_to_samplerate(self, new_samplerate=48000, export=False):
        """Function to downsample the file."""

        # load data
        data, samplerate = sf.read(self.file_path)

        # resample data
        number_of_samples = round(len(data) * float(new_samplerate) / samplerate)
        data_downsampled = signal.resample(data, number_of_samples)

        if export:
            # write to a new file
            new_file_path = self.file_path.replace(".flac", f"_{int(new_samplerate)}Hz.flac")
            sf.write(
                new_file_path,
                data_downsampled,
                new_samplerate,
            )

            self.frequency = new_samplerate
            self.file_path = new_file_path

            with sf.SoundFile(self.file_path) as file:
                logging.info(
                    f"Converting byte count ({self.byte_count}) and byte rate ({self.byte_rate})"
                )

                # set new byte count and byte rate
                log = file.extra_info.split()
                self.byte_count = int(log[log.index("Length") + 2])
                self.byte_rate = self.byte_count / self.file_duration * 8

                logging.info(
                    f"New byte count = {self.byte_count}, new byte rate = {self.byte_rate}"
                )

            logging.info(f"Your file has been downsampled from {samplerate} to {new_samplerate} Hz")


# %% Util functions for working with several files
def read_wav(fullfilepath, start_datetime):
    """Function to read wav file and connect time to it.

    This function returns a dataframe with date and value columns, where
    date columns start at 'start_datetime' and uses frequency from the wav_file
    and value column contains the values from the wav_file."""

    # initialize wav_dataframe
    wav_dataframe = pd.DataFrame(columns=["date", "value"])

    # read wav_file
    wav_array, wav_frequency = sf.read(fullfilepath)

    # make time array starting at start_datetime and using wav_frequency
    time_array = pd.date_range(
        start=start_datetime,
        periods=len(wav_array),
        freq=pd.Timedelta(seconds=1 / wav_frequency),
    )

    # fill dataframe
    wav_dataframe["date"] = time_array
    wav_dataframe["value"] = wav_array

    return wav_dataframe
