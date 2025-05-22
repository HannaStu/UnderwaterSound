"""
Module for System class

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

# External modules
import numpy as np
import pandas as pd

# Internal modules
from soundsignature.file import File


class System:

    def __init__(
        self,
        uid: int,
        recorder: str,
        recorder_number: int,
        no_channels: int,
        amplification_gain: np.array,
        start_date: dt.datetime,
        end_date: dt.datetime,
        system_frequency: int,
        hydrophone: str,
        hydrophone_number: int,
        sensitivity: np.array,
        depth: float,
        noise_floor: float,
        files: list[File],
    ) -> None:
        """
        This class contains information about the system utilised for recording
        underwater noise. A station may have several recording systems (a recorder/hydrophone
        pair with respective parameters gain, sensitivity and depth), which record several audio
        data (name, initial recording time, if annotated, labels, and any comments).

        The required information of a system is documented in the report [D3 Documentation of Standards chosen.pdf]
        which is a deliverable of the CINEA project [CINEA/CD(2022)5010/PP/SI2.899121]:
           > Constructing an open library containing a curated and
           > continuously growing digital catalogue of individual sound
           > signatures from the marine underwater soundscape in
           > shallow seas

        Parameters
        ----------
        recorder : str
            Recorder/data logger type e.g. "SoundTrap".
        recorder_number : int
            Recorder serial number.
        no_channels : int
            Total number of channels of the recorder / datalogger [-].
        amplification_gain : np.array
            Recorder amplification in decibel [dB].
        system_frequency : int
            Original sampling frequency from recorder in Hertz [Hz].
        hydrophone : str
            Description of the manufacturer and the used hydrophone type/model
            e.g. 'Brüell&Kjaer 8106'.
        hydrophone_number : int
            Hydrophone serial number.
        sensitivity : np.array
            Hydrophone sensitivity in decibel reference 1 microPascal [dB re 1V/uPa]
        depth : float
            Position of the hydrophone, in meters below mean sea level [MSL].
        noise_floor : float
            Noise floor of the recorder + hydrophone system [dB re 1uPa]
        files : list
            List of soundsignature.Files associated to this System.
        """
        self.uid = uid
        self.recorder = recorder
        self.recorder_number = recorder_number
        self.no_channels = no_channels
        self.amplification_gain = amplification_gain
        self.start_date = start_date
        self.end_date = end_date
        self.system_frequency = system_frequency
        self.hydrophone = hydrophone
        self.hydrophone_number = hydrophone_number
        self.sensitivity = sensitivity
        self.depth = depth
        self.noise_floor = noise_floor
        self.files = files

    def __repr__(self):
        return f"System {self.hydrophone_number} [{self.start_date}] "

    def read_files(self):
        """Function to read all files."""
        # for file in self.files:
        #     file.read_file()
        raise NotImplementedError("Not implemented yet.")

    @property
    def has_instrument_type(self):
        """Check if type of instrument of current system is known"""
        if type(self.recorder) == str:
            return True
        elif np.isnan(self.recorder) or self.recorder is None:
            return False

    @property
    def has_sensitivity(self):
        """Check if sensitivity of current system is known"""
        if np.isnan(self.sensitivity):
            return False
        else:
            return True

    @property
    def has_gain(self):
        """Check if gain of current system is known"""
        if np.isnan(self.amplification_gain):
            return False
        else:
            return True

    @property
    def known_depth(self):
        """Check if depth of current system is known"""
        if np.isnan(self.depth):
            return False
        else:
            return True

    @property
    def check_dict(self):
        """Property to return metadata quality of current system"""
        return {
            "has_instrument_type": self.has_instrument_type,
            "has_gain": self.has_gain,
            "known_depth": self.known_depth,
            "has_sensitivity": self.has_sensitivity,
        }

    def severity_check(self, test, severity_of_tests):
        """If a test failed, print warning or error based on severity."""
        if severity_of_tests[test] == "high":
            print(
                f"ERROR: station [{self.recorder_number}] failed {test} with high severity!"
            )
        elif severity_of_tests[test] == "low":
            print(
                f"WARNING: station [{self.recorder_number}] failed {test} with low severity!"
            )
        else:
            print(
                f"ERROR: station [{self.recorder_number}] has incorrect severity: [{severity_of_tests[test]}], "
                "please use 'low' or 'high'."
            )

    @classmethod
    def from_frame(cls, systemID, systems_dataframe, file_list):
        """Function to return the system associated to systemID."""

        # Get particular, single row, according to systemID
        system_row = systems_dataframe.loc[
            systems_dataframe["SystemID"] == systemID
        ].squeeze()

        # To verify that station row is just a single row / pd.Series
        if not isinstance(system_row, pd.Series):
            ValueError(f"Multiple rows selected by station ID [{systemID}]")

        # Parse all information of dataframe towards initialization of system object
        return cls(
            recorder=system_row["Recorder"],
            recorder_number=system_row["Recorder no."],
            no_channels=system_row["N-channels"],
            amplification_gain=system_row["amplification gain"],
            start_date=system_row["StartDate system"],
            end_date=system_row["EndDate system"],
            system_frequency=system_row["frequency"],
            hydrophone=system_row["hydrophone"],
            hydrophone_number=system_row["hydrophone number"],
            sensitivity=system_row["sensitivity"],
            depth=system_row["depth"],
            noise_floor=system_row["noise floor"],
            files=file_list,
        )

    @classmethod
    def from_series(cls, system_row):
        # Parse all information of dataframe towards initialization of system object
        return cls(
            uid=system_row["SystemID"],
            recorder=system_row["Recorder"],
            recorder_number=system_row["Recorder no."],
            no_channels=system_row["N-channels"],
            amplification_gain=system_row["amplification gain"],
            start_date=system_row["StartDate system"],
            end_date=system_row["EndDate system"],
            noise_floor=system_row["noise_floor"],
            system_frequency=system_row["frequency"],
            hydrophone=system_row["hydrophone"],
            hydrophone_number=system_row["hydrophone number"],
            sensitivity=system_row["sensitivity"],
            depth=system_row["depth"],
            files=[],
        )

    def to_dict(self) -> dict:
        """Function to return dictionary of the system"""

        return {
            "uid": self.uid,
            "Recorder": self.recorder,
            "Recorder no.": self.recorder_number,
            "N-channels": self.no_channels,
            "amplification gain": self.amplification_gain,
            "startdate": self.start_date,
            "enddate": self.end_date,
            "noise floor": self.noise_floor,
            "system frequency": self.system_frequency,
            "hydrophone": self.hydrophone,
            "hydrophone number": self.hydrophone_number,
            "sensitivity": self.sensitivity,
            "depth": self.depth,
        }
