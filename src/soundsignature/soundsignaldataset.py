"""
Module for SoundSignalDataset class

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
import sys
import shutil

#import cartopy.crs as ccrs
import librosa
import matplotlib.pyplot as plt
import numpy as np

# External modules
import pandas as pd
import soundfile as sf

PARENT_PROJECT_FOLDER = r"C:\Users\hurh\GitHub\underwater-noise\src"
sys.path.append(PARENT_PROJECT_FOLDER)


import soundsignature.utilities as utl
from soundsignature.annotation import LIST_VALID_SOURCES, Annotation
from soundsignature.extract_meta import read_metadata
from soundsignature.file import File

# Internal modules
from soundsignature.station import Station
from soundsignature.system import System


class SoundSignalDataset:

    def __init__(
        self, uid: int, name: str, folder: str, stations: list[Station]
    ) -> None:
        """This class contains information from one dataset utilised for recording
        underwater noise. A SoundSignalDataset may have several stations, which may have several
        recording systems (a recorder/hydrophone pair with respective parameters gain, sensitivity
        and depth), which record several audio data (name, initial recording time, if annotated,
        labels, and any comments).

        Parameters
        ----------
        uid : int
            Unique identifier of the SoundSignalDataset. This is commonly determined in a meta data excel sheet.
        name : str
            Name of the SoundSignalDataset.
        folder : str
            Path to the subfolder of the dataset, relative from the globally defined DATADIR env-variable.
            e.g. ./borssele-rws-nl/ for the borssele dataset. This folder definition is commonly defined in a Meta data excel sheet.
        stations : list[Station]
            List of soundsignature. Stations associated to this SoundSignalDataset.

        """
        self.uid = uid
        self.name = name
        self.folder = folder
        self.stations = stations
        self.quality = None

    def __repr__(self):
        return f"Dataset [{self.name}]"

    @property
    def files(self):
        """Function to get a list of all files within the current dataset."""
        list_of_files = []

        # Files of a dataset are all the files of all the underlying stations
        for station in self.stations:
            list_of_files += station.files

        return list_of_files

    @property
    def total_duration(self):
        """Function to calculate the total sound recording duration of the current dataset in seconds."""

        durations = [file.file_duration for file in self.files]

        return sum(durations)

    def load_from_ftp(self):
        """Function to get the data from the database.

        Connected to issue #10
        https://github.com/soundsignature/WP2_T3/issues/10
        """

        raise NotImplementedError("Not implemented yet.")

    def show_map(self):
        """Function for issue #7 https://github.com/soundsignature/WP2_T3/issues/7

        Goal of this function:
        Point all coordinates (WGS84) of the underlying stations of a single SoundSignalDataset on a map.
        A base-map is needed for orientation, lines will be sufficient. Use cartopy or equivalent.
        """
        # get base map
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines("auto")

        # for each station, plot a dot with latitude/longitude and name on map
        for station in self.stations:
            ax.plot(
                station.longitude,
                station.latitude,
                color="blue",
                linewidth=1,
                marker="o",
            )
            ax.text(station.longitude + 0.01, station.latitude + 0.01, station.name)

        # Addition: use of extents of Europe:
        # (left, right, bottom, top)
        ax.set_extent([-15, 35, 30, 75])

        fig.tight_layout()

        return fig, ax

    def file_density(self):
        """Function for issue #9 https://github.com/soundsignature/WP2_T3/issues/9

        Goal of this function:
        List all files of the systems of the stations of the SoundSignalDataset.
        Plot a density diagram with time on x-axis, and <system>-<station> on y-axis."""

        # initialize intermediate output
        # TO DO: Proper initialization: all information about stations and systems are known!
        intermediate_output = pd.DataFrame()

        # loop over stations
        for station in self.stations:

            # loop over systems of this station
            for system in station.systems:

                # get name for current station-system combination
                stations_system_name = (
                    f"{station.name} - {system.hydrophone_number:.0f}"
                )

                # initialize dictionary for current station
                station_system_density = {}

                # if station has no files, fill column with zeroes
                if len(system.files) == 0:
                    intermediate_output[stations_system_name] = 0
                    continue

                else:
                    # loop over all files of current station
                    for file in system.files:

                        # from file path, get file name and extract date from it
                        filename = os.path.basename(file.file_path)
                        filedatestring = "20" + filename.split(".")[1]
                        filedate = filedatestring[:8]

                        # make datetime from datestring
                        date_of_file = dt.datetime(
                            year=int(filedate[0:4]),
                            month=int(filedate[4:6]),
                            day=int(filedate[6:8]),
                        )

                        # fill station_system_density dictionary with amount of files per date
                        if date_of_file in station_system_density:
                            station_system_density[date_of_file] += 1
                        else:
                            station_system_density[date_of_file] = 1

                    # convert current station density to station column
                    current_station_df = pd.DataFrame.from_dict(
                        station_system_density,
                        orient="index",
                        columns=[stations_system_name],
                    )

                    # join with intermediate_output
                    intermediate_output = pd.concat(
                        [intermediate_output, current_station_df], axis=0
                    )
        if intermediate_output.empty:
            print(
                f"ERROR: Dataset [{self.name}] does not have any files in the metadata, "
                "or no date in the file name. Please check this."
            )
            fig, ax = plt.subplots()
            return fig, ax

        # resample to get all possible days from min_date to max_date and reset index
        resampled_output = intermediate_output.resample("d").sum()
        resampled_output.reset_index(inplace=True)

        # add additional column with "YYYY-MM-01" format
        resampled_output["YearMonth"] = pd.to_datetime(resampled_output["index"]).apply(
            lambda x: dt.datetime(x.year, x.month, 1)
        )

        # list of all station-systems combinations
        all_stations_systems = resampled_output.columns
        all_stations_systems = all_stations_systems.drop(["index", "YearMonth"])

        # new dataframe grouped by 'year-month-01' combination
        # only taking the columns of all_stations_systems
        total_dataframe = resampled_output.groupby("YearMonth")[
            all_stations_systems
        ].sum()

        # plot density chart
        fig, ax = utl.plotDensityChart(total_dataframe)

        # new title
        ax.set_title(f"Density of files of dataset: {self.name}")

        # # get datelabels in "YYYY-MM" template
        datelist = total_dataframe.index.tolist()
        datelabels = [d.strftime("%Y-%m") for d in datelist]
        ax.set_xticks(range(0, len(datelabels)))
        ax.set_xticklabels(datelabels, rotation=90)

        fig.set_size_inches(10, 8, forward=True)
        fig.tight_layout()

        return fig, ax

    def sample_density(self):
        """Function for later issue"""

        raise NotImplementedError("Not implemented yet.")

    def meta_check_results(self):
        """For current dataset, evaluate the associated data.
        The following tests will be done:
        - coordinates are within EU region (on each station)
        - water depth is shallow (on each station)
        - instrument type is known (on each system)
        - sensitivity is known (on each system)
        - gain is known (on each system)
        - monitoring depth is known (on each system)
        """
        print(f"INFO: Starting on evaluation on dataset [{self.name}]")
        print(f"INFO: [{self.name}] has {len(self.stations)} stations.")

        # severity of above-mentioned tests
        severity_of_system = {
            "has_instrument_type": "high",
            "has_sensitivity": "high",
            "has_gain": "low",
            "known_depth": "low",
        }

        severity_of_station = {
            "in_europe": "high",
            "in_shallow_water": "low",
        }

        # dict with results
        result_dict = {}
        recommendations = []
        total_tests = 0
        passed_tests = 0

        for station in self.stations:
            print(f"INFO: Evaluation of station {station.name}")

            # per station, add two tests
            total_tests += 2

            # per system, add four tests
            total_tests += 4 * len(station.systems)

            result_dict[station] = station.check_dict

            if not station.in_europe:
                recommendations.append(
                    f"Station [{station}] is not in Europe, consider checking coordinates"
                )
            if not station.in_shallow_water:
                recommendations.append(
                    f"Station [{station}] is not in shallow water, consider checking depth"
                )

            # count passed tests
            for system in station.systems:
                system_dict = station.check_dict[system]
                passed_tests += sum(system_dict.values())

                if not system.has_instrument_type:
                    recommendations.append(
                        f"System [{system}] has no instrument type, please check metadata"
                    )
                if not system.has_sensitivity:
                    recommendations.append(
                        f"System [{system}] has no sensitivity, please check metadata"
                    )
                if not system.has_gain:
                    recommendations.append(
                        f"System [{system}] has no gain, please check metadata"
                    )
                if not system.known_depth:
                    recommendations.append(
                        f"System [{system}] has no depth, please check metadata"
                    )

                for system_test in severity_of_system:
                    if not system_dict[system_test]:
                        print(
                            f"System [{system}] failed [{system_test}] with {severity_of_system[system_test]} severity"
                        )

            passed_tests += station.in_europe
            passed_tests += station.in_shallow_water

            for station_test in severity_of_station:
                if not station.check_dict[station_test]:
                    print(
                        f"Station [{station}] failed [{station_test}] with {severity_of_station[station_test]} severity"
                    )

        fraction_passed = round(passed_tests / total_tests, 2)
        print(f"Test percentage of dataset [{self.name}] is {fraction_passed*100}%")
        self.quality = fraction_passed * 100
        return result_dict, recommendations

    def check_data_results(self) -> pd.DataFrame:
        """Function to get all signal-to-noise ratios + entropy values for all files in current dataset."""

        # initialise dataframe with results
        quality_parameters = []
        # quality_recommendations = []

        # loop over all stations
        for station in self.stations:

            logging.info(f"Working on station [{station}]")

            # loop over all systems
            for system in station.systems:

                logging.info(f"Working on system [{system}]")
                # system information needed to calculate SNR
                noise_floor = system.noise_floor
                sensitivity = system.sensitivity
                gain = system.amplification_gain

                # calculate SNR and entropy for every file
                for file in system.files:

                    logging.info(f"Working on [{file}]")
                    result = {"File": file.file_path}

                    logging.info("Calculating SNR and entropy")
                    (
                        result["SNR (99 percentile)"],
                        result["SNR (rms)"],
                        result["entropy"],
                    ) = file.calculate_file_snr_entropy(noise_floor, sensitivity, gain)

                    result["frequency"] = file.frequency
                    result["file durations [s]"] = file.file_duration
                    result["label types"] = [x.label_type for x in file.annotations]
                    result["label sources"] = [x.label_source for x in file.annotations]

                    logging.info(f"Done with file [{file}]")
                    # add result to total list of results
                    quality_parameters.append(result)

                    # # check the results and give recommendations
                    # quality_recommendation = {
                    #     "File": [],
                    #     "frequency": [],
                    #     "SNR": [],
                    #     "entropy": [],
                    #     "sample durations [s]": [],
                    #     "label types": [],
                    #     "label sources": [],
                    # }

                    # # frequency should be smaller or equal to 48 kHz
                    # if result["frequency"] > 48000:
                    #     quality_recommendation["frequency"] = (
                    #         "Sampling rate of {} kHz is bigger than 48 kHz".format(
                    #             result["frequency"] / 1000
                    #         )
                    #     )
                    #
                    # # SNR constraint
                    # if result["SNR (rms)"] < 0:
                    #     quality_recommendation["SNR"] = (
                    #         "SNR of {} dB is smaller than 0 dB".format(
                    #             result["SNR (rms)"]
                    #         )
                    #     )
                    #
                    # # entropy constraint
                    # if result["entropy"] > 0.8:
                    #     quality_recommendation["entropy"] = (
                    #         "Entropy of {} is bigger than 0.8".format(result["entropy"])
                    #     )
                    #
                    # # sample duration of label should not be longer than 10 minutes
                    # if not all(x < 10 * 3600 for x in result["sample durations"]):
                    #     quality_recommendation["sample duration"] = (
                    #         "File contains annotation longer than 10 min"
                    #     )
                    #
                    # # check label type
                    # if not all(x.has_valid_type_label() == True for x in file.annotations):
                    #     quality_recommendation["label types"] = (
                    #         "File contains annotation(s) with invalid label type"
                    #     )
                    #
                    # # check label source
                    # if not all(x.has_valid_source_label() == True for x in file.annotations):
                    #     quality_recommendation["label sources"] = (
                    #         "File contains annotation(s) with invalid label source"
                    #     )
                    #
                    # # add the recommendations to the dataframe
                    # if any(quality_recommendation.values()):
                    #     quality_recommendation["File"] = file.file_path
                    #     df_quality_recommendations.append(quality_recommendation)

        # convert to dataframe
        df_quality_parameters = pd.DataFrame(quality_parameters)
        # df_quality_recommendations = pd.DataFrame(quality_recommendations)

        return df_quality_parameters  # , df_quality_recommendations

    def data_quality_histogram(self):
        """Function to plot the SNR and entropy of all files in an histogram."""

        # get dataframe with SNR and entropy
        quality_results = self.check_data_results()
        snr_rms = quality_results["SNR (rms)"]
        snr_99 = quality_results["SNR (99 percentile)"]
        entropy = quality_results["entropy"]

        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 8))

        # histogram of SNR values
        ax[0].hist(snr_rms, bins=10)
        ax[0].set_xlabel("SNR (rms)")
        ax[0].set_ylabel("amount of files")
        ax[0].grid()

        # histogram of SNR values
        ax[1].hist(snr_99, bins=10)
        ax[1].set_xlabel("SNR (99 percentile)")
        ax[1].set_ylabel("amount of files")
        ax[1].grid()

        # histogram of entropy values
        ax[2].hist(entropy, bins=10)
        ax[2].set_xlabel("entropy (normalized)")
        ax[2].set_ylabel("amount of files")
        ax[2].grid()

        plt.suptitle(
            "Histogram of SNR and entropy in the files in the dataset \n"
            "Dataset: {}".format(self.name)
        )
        plt.tight_layout()

        return fig, ax, quality_results

    def check_integrity_metadata(self):
        """Function to check the integrity of the metadata of the dataset.

        Main check is that a hydrophone number is not used in multiple stations at the same time.
        """

        list_of_systems = []

        # check if one system is not used in multiple stations at the same time
        for station in self.stations:
            for system in station.systems:
                list_of_systems.append(system)

        # get list of unique hydrophone number of systems
        unique_hydrophone_numbers = set(
            [system.hydrophone_number for system in list_of_systems]
        )

        for hydrophone_number in unique_hydrophone_numbers:
            print(f"Checking hydrophone number [{hydrophone_number}]")
            systems_with_same_hydrophone_number = [
                system
                for system in list_of_systems
                if system.hydrophone_number == hydrophone_number
            ]
            # Shortcut: if it is only one, then we're done.
            if len(systems_with_same_hydrophone_number) < 2:
                # If hydrophone is used once, then it's not doubled at all.
                continue

            # QC BOEJ7: volgens mij kan onderstaand een stuk netter. Voor nu: low-prio.
            # check if systems in systems_with_same_hydrophone_number have no overlap in start and end time
            for index_1, system in enumerate(systems_with_same_hydrophone_number):
                for index_2, other_system in enumerate(
                    systems_with_same_hydrophone_number
                ):
                    if index_1 != index_2:
                        if (
                            system.start_date < other_system.end_date
                            and system.end_date > other_system.start_date
                        ):
                            raise ValueError(
                                f"ERROR: hydrophone number [{hydrophone_number}] is used in multiple stations at the same time"
                            )

    @classmethod
    def from_excel(cls, dataset_id: int, path_excel: str):
        """Function to read the Excel and get the dataset and all children from dataset_id."""

        # Initiate excel-object
        xl = pd.ExcelFile(path_excel)

        # Check metadata_sheet integrity
        # It should contain at least three sheets, with given names
        mandatory_sheetnames = {"SoundSignalDataset", "Station", "System"}
        if not mandatory_sheetnames.issubset(xl.sheet_names):
            logging.info(f"Not all mandatory sheets were found in the Excel")

            # Investigate the difference, to get the sheets which are mandatory but not in the sheet.
            missing_sheets = mandatory_sheetnames.difference(xl.sheet_names)
            logging.info(f"The following sheets are missing: {missing_sheets}")
            raise KeyError(
                f"The meta data excelfile does not contain sheet(s): {missing_sheets}"
            )

        # read dataset-sheet from metadata in 'path'
        soundsignaldataset_frame = xl.parse(sheet_name="SoundSignalDataset")
        xl.close()

        # get info from current dataset (1 row)
        dataset_row = soundsignaldataset_frame.loc[
            soundsignaldataset_frame["SoundSignalDatasetID"] == dataset_id
        ].squeeze()

        # To verify that station row is just a single row / pd.Series
        if len(dataset_row) == 0:
            raise ValueError(
                f"No information available in excelsheet for dataset ID [{dataset_id}]"
            )
        if not isinstance(dataset_row, pd.Series):
            raise ValueError(f"Multiple rows selected by dataset ID [{dataset_id}]")

        # Build the class, which (currently) an empty list of stations
        dataset = cls(
            uid=dataset_id,
            name=dataset_row["Name"],
            folder=dataset_row["Folder"],
            stations=[],
        )

        # %% Continue with underlying stations (if available)
        station_frame = pd.read_excel(path_excel, sheet_name="Station")

        ## THIS SECTION SHOULD BE PUT INTO METHOD: def add_stations(self, station_frame)
        # get all station data for current dataset (0, 1 or more rows)
        station_rows = station_frame.loc[
            station_frame["SoundSignalDatasetID"] == dataset.uid
        ]
        logging.info(
            f"{len(station_rows)} station(s) available for dataset [{dataset.name}]"
        )

        # add the stations to the current dataset (with each station an empty list of systems)
        dataset.stations = [
            Station.from_series(row) for _, row in station_rows.iterrows()
        ]

        # %% Continue with underlying systems (if available) per station
        system_frame = pd.read_excel(path_excel, sheet_name="System")
        for station in dataset.stations:
            # get all system data for current station (0, 1 or more rows)
            system_rows = system_frame.loc[system_frame["StationID"] == station.uid]
            logging.info(
                f"{len(system_rows)} systems(s) available for station [{station.name}]"
            )

            # add the systems to the current station
            station.systems = [
                System.from_series(row) for _, row in system_rows.iterrows()
            ]

        # %% Continue with attached files, if available and if sheet exists.
        try:
            file_frame = pd.read_excel(path_excel, sheet_name="File")
            # add datadir to file paths
            file_frame["Path"] = file_frame["Path"].apply(
                lambda x: os.path.join(os.getenv("DATADIR"), x)
            )
        except ValueError:
            logging.info('Sheet "File" not in Excel-file, done with reading')
            return dataset

        logging.info(
            f"{len(file_frame)} files(s) available for dataset [{dataset.name}]"
        )
        if len(file_frame) == 0:
            logging.info(f"Done with reading")
            return dataset

        # If files are present in given sheet, then add that files towards the corresponding (parent) system
        for station in dataset.stations:
            for system in station.systems:
                # get all file data for current system (0, 1 or more rows)
                file_rows = file_frame.loc[file_frame["SystemID"] == system.uid]
                logging.info(
                    f"{len(file_rows)} file(s) available for system [{system.recorder}]"
                )
                # glue the selected files to the current system
                system.files = [
                    File.from_series_only_path(row) for _, row in file_rows.iterrows()
                ]

        logging.info(f"Done with reading")

        return dataset

    def append_files_from_folder(self, extension=".flac", list_of_files=None):
        """
        Function to add File objects, based on files in a folder.
        It is assumed that the dataset is already created.

        Parameters
        ----------
        extension : str, optional
            Extension of the files to be added. The default is ".flac".
        list_of_files : list, optional
            List of filenames to select a proper subset of all files in the folder of the dataset. The default is None.

        """

        # if files are in metadata, do not append them from the folder (NON-DUTCH ONLY)
        if len(self.files) > 0:
            logging.warning(
                f"Dataset {self.name} already has {len(self.files)} files from the metadata"
            )
            logging.warning("Files will not be appended from the folder")
            return

        # Get absolute path of the dataset folder. Hence, dataset_dir in the calling methods isn't needed any more.
        dataset_folder = os.path.join(os.getenv("DATADIR"), self.folder)
        # The designated folder of the dataset should exist.
        if not os.path.exists(dataset_folder):
            raise FileNotFoundError(f"Dataset folder {dataset_folder} does not exist")

        # List all (full)filenames with in dataset folder
        dataset_audio_files = utl.getFilePaths(dataset_folder, extension=extension)

        if not dataset_audio_files:
            logging.warning(
                f"No audio files found in folder [{dataset_folder}] with extension [{extension}]"
            )
            return

        # only add files that are in the (optional) list_of_files
        if list_of_files:
            # Selecting, based on the BASENAME of the audio files found in dataset folder.
            dataset_audio_files = [
                fullfilename
                for fullfilename in dataset_audio_files
                if os.path.basename(fullfilename) in list_of_files
            ]

        logging.info(
            f"Found {len(dataset_audio_files)} audio files in folder {dataset_folder}"
        )
        num_coupled_files = 0

        # make a list of stations for current dataset
        for station in self.stations:
            logging.info(f"Working on station: [{station}]")
            for system in station.systems:
                logging.info(f"Working on system: [{system}]")

                # for dutch datasets, add the waterproof files
                if "NL" in self.name:
                    system, num_coupled_files = self.add_dutch_file_to_system(
                        system,
                        dataset_audio_files,
                        num_coupled_files,
                        extension,
                    )
                else:
                    raise NotImplementedError(
                        "Only Dutch datasets are supported for now"
                    )

        logging.info(
            f"Found [{len(dataset_audio_files)}] audio files, added [{num_coupled_files}] to systems"
        )
        # check which files did not get added
        coupled_files = [file.file_path for file in self.files]
        not_coupled_files = [
            file for file in dataset_audio_files if file not in coupled_files
        ]

        # Only a warning if there are any non-coupled files
        if not_coupled_files:
            logging.warning("Files not coupled:")
            for file in not_coupled_files:
                logging.warning(f" - {file}")

        return

    def add_dutch_file_to_system(
        self, system, dataset_audio_files, num_coupled_files, extension=".flac"
    ):
        """
        Add Dutch files to the system. The file names contains the hydrophone number and the date.

        The Dutch files are named as follows:
        - {hydrophone_number}.{datestring}.{extension}
            where hydrophone_number can be the pure hydrophone number or the hydrophone number with a prefix.
            i.e. 1208774689 for hydrophone number 4689 or 5397 for hydrophone number 5397.
        - {datestring} is the date of the recording in the format YYMMDDHHMM
        - {extension} is the file extension (default .flac)

        Files will get coupled to the system if the date of the file lies within the start and
        end date of the system and when the hydrophone number is in the filename.


        Args:
            system (System): The system to which the files will be added.
            dataset_audio_files (list): List of audio file paths in the dataset.
            num_coupled_files (int): The number of files already coupled to the system.
            extension (str, optional): The file extension to consider. Defaults to ".flac".

        Returns:
            system: The updated system with the added files.
            num_coupled_files: The updated number of coupled files.
        """

        fullfilenames = [
            fullfilename
            for fullfilename in dataset_audio_files
            if str(int(system.hydrophone_number))
            in os.path.basename(fullfilename).split(".")[0]
        ]

        if not fullfilenames:
            logging.warning(
                f"No files found for hydrophone number [{system.hydrophone_number}]"
            )
            return system, num_coupled_files

        # list of date strings of 'files'
        files_date_string = [
            f'20{fullfilename.split(f"{int(system.hydrophone_number)}.")[1][:10]}'
            for fullfilename in fullfilenames
        ]

        # check if date of file lies in system interval
        check_start_date = system.start_date <= pd.to_datetime(files_date_string)
        check_end_date = pd.to_datetime(files_date_string) <= system.end_date
        in_date_range = check_start_date * check_end_date

        # only the files lying within the daterange of current system
        fullfiles_within_daterange = [
            fullfile
            for (fullfile, in_date) in zip(fullfilenames, in_date_range)
            if in_date
        ]

        # append the files belonging to this system to 'list_of_files'
        for fullfile in fullfiles_within_daterange:
            # corresponding metadatafile should exist
            metadatafile = fullfile.replace(extension, ".log.xml")
            if not os.path.exists(metadatafile):
                logging.warning(
                    f"Metadatafile [{metadatafile}] does not exist, skipping"
                )
                continue

            _, file_info, _ = read_metadata(metadatafile, audio_extension=extension)
            # If (after reading) metadata is not available, skip adding the file
            if not file_info:
                logging.warning(f"File [{fullfile}] has no metadata, skipping")
                continue

            file_info["file_path"] = fullfile
            file_info["compressed"] = "No"  # standard?
            file_info["annotations"] = []  # To be added later

            # based on file info, make a file instance.
            file_obj = File(**file_info)

            # append file to list_of_files
            system.files.append(file_obj)
            num_coupled_files += 1

            # To give every 10 files a response
            if num_coupled_files % 10 == 0:
                logging.info(f"Number of coupled files: {num_coupled_files}")

        return system, num_coupled_files

    def append_labels(self, process_dict_path, path_label_excel):
        """
        Appends labels to the correct files from the dataset based on the label Excel file.
        Uses the provided process_dict as additional information for the labels.

        Args:
            process_dict_path (str): The file path of the process_dict Excel file.
            path_label_excel (str): The file path of the label Excel file.

        """
        # read process_dict excel from path
        process_dict = pd.read_excel(process_dict_path)

        # check if label file is in process_dict
        if os.path.basename(path_label_excel) not in list(
            process_dict["annotation_file"]
        ):
            logging.error(
                f"Label file [{path_label_excel}] is not in the process_dict [{process_dict_path}]"
            )
            return

        # read dataset-sheet with labelling data
        label_frame = pd.read_excel(path_label_excel).drop_duplicates()

        # get process dict for this label file
        process_dict = (
            process_dict.loc[
                process_dict["annotation_file"] == os.path.basename(path_label_excel)
            ]
            .squeeze()
            .to_dict()
        )

        # check if all columns are present
        obliged_columns = {
            "Audio Name",
            "Time min (s)",
            "Time max (s)",
            "Type Label",
            "Source Label",
            "Frequency min (Hz)",
            "Frequency max (Hz)",
        }
        if not obliged_columns.issubset(label_frame.columns):
            logging.error(
                f"Label frame from path {path_label_excel} does not contain all necessary columns"
            )
            return

        # rename all audio name file to the basename
        label_frame["Audio Name"] = label_frame["Audio Name"].apply(
            lambda x: os.path.basename(x)
        )

        for station in self.stations:
            for system in station.systems:
                for file in system.files:
                    file_name = os.path.basename(file.file_path)
                    # Selecting all rows of the excel file which are related to the particular audio file
                    file_rows = label_frame.loc[label_frame["Audio Name"] == file_name]

                    for _, row in file_rows.iterrows():
                        file.annotations.append(
                            Annotation.from_row(file, process_dict, row)
                        )

        # extra check that annotations parent_file must match file
        for file in self.files:
            file.annotations = [
                annotation
                for annotation in file.annotations
                if annotation.parent_file == file
            ]

        logging.info(f"Labels have been added to the dataset from [{path_label_excel}]")

        return

    def get_labels_of_source(self, label_source: str):
        """Function to get all labels of a certain source in the dataset."""

        # check if label_source is in list_valid_sources
        if label_source not in LIST_VALID_SOURCES:
            logging.error(f"Label source [{label_source}] is not valid.")
            return
        else:
            logging.info(f"Getting all labels of source [{label_source}]")

        # make a list of all annotations of the current dataset
        all_source_labels = []
        for file in self.files:
            for annotation in file.annotations:
                # check if label_source is in annotation.label_source after splitting on '|'
                if label_source in annotation.label_source.split("|"):
                    all_source_labels.append(annotation)

        return all_source_labels

    def downsample_dataset(self, new_samplerate=48000):
        """Function to downsample all files in the dataset."""
        for file in self.files:
            logging.info(f"Working on [{file}]")

            if file.frequency > new_samplerate:
                file.downsample_to_samplerate(new_samplerate)

        logging.info(f"Your files have been downsampled to {new_samplerate} Hz")

    def to_excel(self, output_dir, annotations_df=pd.DataFrame()):
        """Function to collect all station, system and file information of the dataset and write it to Excel."""

        # make a list for every tab
        df_sound_signal_dataset = []
        df_station = []
        df_system = []
        df_file = []
        df_annotations = []

        # make the dataset tab
        sound_signal_dataset_dict = {}
        sound_signal_dataset_dict["SoundSignalDatasetID"] = self.uid
        sound_signal_dataset_dict["Name"] = self.name
        sound_signal_dataset_dict["metadata quality"] = f"{self.quality} %"

        df_sound_signal_dataset.append(sound_signal_dataset_dict)

        # loop over all stations, sytems and files
        file_id = 1
        annotation_id = 1

        # loop over all stations within the dataset
        for station in self.stations:

            # define station ID corresponding to dataset ID
            station_dict = {"StationID": station.uid, "SoundSignalDatasetID": self.uid}

            # get additional info from one line in the excel file: information about the station
            station_dict.update(station.to_dict())
            df_station.append(station_dict)

            # loop over all systems for the station
            for system in station.systems:

                # define system ID corresponding to station ID
                system_dict = {"SystemID": system.uid, "StationID": station.uid}

                # get additional info from one line in the excel file: information about the system
                system_dict.update(system.to_dict())
                df_system.append(system_dict)

                # loop over all files of a system
                for file in system.files:

                    # define file ID corresponding to system ID
                    file_dict = {"FileID": file_id, "SystemID": system.uid}

                    # get one line in the excel file: information about the file
                    file_dict.update(file.to_dict())

                    # remove annotations, snr and entropy from file
                    file_dict.pop("annotations")
                    file_dict.pop("snr")
                    file_dict.pop("entropy")

                    df_file.append(file_dict)

                    # add annotations as a tab if they are given
                    if annotations_df.empty:
                        continue
                    else:
                        annotations = annotations_df.loc[
                            annotations_df["Original audio file"]
                            == os.path.relpath(file.file_path, os.getenv("DATADIR"))
                        ]

                        for _, annotation in annotations.iterrows():
                            annotation_dict = {
                                "AnnotationID": annotation_id,
                                "FileID": file_id,
                            }
                            annotation_dict.update(annotation.to_dict())
                            df_annotations.append(annotation_dict)

                            annotation_id += 1

                    file_id += 1

        # Create dataframe
        df_sound_signal_dataset = pd.DataFrame(df_sound_signal_dataset)
        df_station = pd.DataFrame(df_station)
        df_system = pd.DataFrame(df_system)
        df_file = pd.DataFrame(df_file)
        df_annotations = pd.DataFrame(df_annotations)

        # write to an excel file
        with pd.ExcelWriter(output_dir, engine="xlsxwriter") as writer:
            df_sound_signal_dataset.to_excel(
                writer, sheet_name="SoundSignalDataset", index=False
            )
            df_station.to_excel(writer, sheet_name="Station", index=False)
            df_system.to_excel(writer, sheet_name="System", index=False)
            df_file.to_excel(writer, sheet_name="File", index=False)
            df_annotations.to_excel(writer, sheet_name="Annotations", index=False)

    def export_samples_for_training(self, annotations, path_export):
        """
        Function to export labelled samples for training
        NOT USED YET, SINCE FULL FILES ARE EXPORTED, WITH export_files_with_annotations()
        """

        # initialize labels
        labels = pd.DataFrame(columns=["Audio name", "Label Source"])

        # keep track of annotation counter
        annotation_counter = 1
        for annotation in annotations:
            logging.info(
                f"Working on annotation {annotation_counter} / {len(annotations)}"
            )
            audio_array, sr = librosa.load(annotation.parent_file.file_path, sr=None)
            tmin = int(annotation.tmin * sr)
            tmax = int(annotation.tmax * sr)
            audio_array = audio_array[tmin:tmax]
            file = os.path.basename(annotation.parent_file.file_path)
            sf.write(
                os.path.join(
                    path_export,
                    f"{annotation_counter}_{file}_{annotation.label_source}.flac",
                ),
                audio_array,
                sr,
            )
            # fill metadata row by row with audio name and label source
            labels.loc[annotation_counter - 1] = [
                f"Annotation_{annotation_counter}.flac",
                annotation.label_source,
            ]

            annotation_counter += 1

        # export metadata to csv
        labels.to_csv(os.path.join(path_export, "labels.csv"), index=False)
        logging.info(
            f"Exported {len(annotations)} samples for training to {path_export}"
        )

    def export_files_with_annotations(self, output_dir):
        """
        Function to export all files with annotations in the dataset,
        export all annotations to a csv file and copy the files to a new folder.

        Parameters
        ----------
        output_dir : str
            The path to the directory where the files and annotations should be exported to.
        """

        # get all files with annotations
        files_with_annotations = [file for file in self.files if file.annotations]
        logging.info(
            f"Found {len(files_with_annotations)} files with annotations in dataset {self.name}"
        )

        # export annotations to a csv file
        annotations = [
            annotation.to_dict()
            for file in files_with_annotations
            for annotation in file.annotations
        ]
        logging.info(f"Found {len(annotations)} annotations in dataset {self.name}")
        annotations_df = pd.DataFrame(annotations)

        # check if annotations are overlapping
        unique_files = annotations_df["parent_file"].unique()

        for file in unique_files:
            file_annotations = annotations_df[annotations_df["parent_file"] == file]
            file_annotations = file_annotations.sort_values(by="tmin")

            # Check for overlaps
            starts = file_annotations["tmin"].values
            ends = file_annotations["tmax"].values

            # make boolean matrix of overlapping intervals
            overlapping = (starts[:, None] < ends) & (ends[:, None] > starts)
            overlapping = overlapping & ~np.eye(
                overlapping.shape[0], dtype=bool
            )  # Remove self-comparison (diagonal of matrix)

            # Mark overlapping intervals
            overlap_indices = overlapping.any(axis=1)

            # Mark overlapping intervals
            for i in range(len(file_annotations)):
                overlap_info = ""
                overlap_indices = np.where(overlapping[i])[0]

                # Note down whether there are overlapping intervals
                if len(overlap_indices) == 0:
                    annotations_df.loc[file_annotations.index[i], "overlapping"] = False
                else:
                    annotations_df.loc[file_annotations.index[i], "overlapping"] = True

                for index in overlap_indices:
                    # get information about the overlap
                    overlapping_start = max(starts[i], starts[index])
                    overlapping_end = min(ends[i], ends[index])
                    overlap_info += f"Overlaps with index {file_annotations.index[index]}, Start: {overlapping_start}, End: {overlapping_end}, "

                annotations_df.loc[file_annotations.index[i], "overlap_info"] = (
                    overlap_info
                )

        # Ensure changes are reflected back in the original DataFrame
        annotations_df.update(annotations_df)

        annotations_df.to_csv(os.path.join(output_dir, "annotations.csv"), sep=";", decimal = ",") #CHANGED
        logging.info(f"Exported annotations.csv to {output_dir}")

        # since the files should be one-to-one copied (no snipping, no resampling, etc.) the files can be copied
        # without modification to output dir
        for file in files_with_annotations:
            if not os.path.exists(
                os.path.join(output_dir, os.path.basename(file.file_path))
            ):
                shutil.copy(file.file_path, output_dir)
                logging.info(f"Copied {file.file_path} to {output_dir}")
        logging.info(f"Copied files to {output_dir}")

    def export_files_for_catalogue(self, output_dir):
        """
        Export audio files and annotations for a catalogue.

        Args:
            output_dir (str): The directory where the exported files and annotations will be saved.
        """

        # get all files with annotations
        files_with_annotations = [file for file in self.files if file.annotations]
        logging.info(
            f"Found {len(files_with_annotations)} files with annotations in dataset {self.name}"
        )

        annotations_df = pd.DataFrame(
            columns=[
                "File",
                "Original audio file",
                "Label Source",
                "Label Type",
                "Time min (s)",
                "Time max (s)",
                "Frequency min (Hz)",
                "Frequency max (Hz)",
            ]
        )
        annotation_index = 1

        datadir = os.getenv("DATADIR")
        # for a few datasets, the files must be read, snipped according to annotation and exported
        for file in files_with_annotations:
            logging.info(f"Working on file {file.file_path}")
            logging.info(f"File has {len(file.annotations)} annotations")
            audio_array, sr = sf.read(file.file_path)

            extension = file.file_path.split(".")[-1]  # reqquired for audio name

            # get zone from station of file, returns None if no station found
            sea_zone = next(
                (station.zone for station in self.stations if file in station.files),
                None,
            )  # required for audio name

            # loop over all annotations in the file, get required information
            for annotation in file.annotations:
                logging.info(f"Working on annotation {annotation_index}")
                duration = round(
                    annotation.tmax - annotation.tmin
                )  # required for audio name

                # # duration of the sample should be larger than 1 second for catalogue
                # if duration < 1:
                #     logging.warning(
                #         f"{annotation} from {annotation.tmin} to {annotation.tmax} in file {file.file_path} is too short."
                #     )

                #     continue

                idx_tmin = int(annotation.tmin * sr)
                idx_tmax = int(annotation.tmax * sr)
                audio_array_annotation = audio_array[idx_tmin:idx_tmax]

                # read byte sample from wav/flac-file
                with sf.SoundFile(file.file_path) as metadata_file:
                    byte_sample = [
                        int(temp)
                        for temp in metadata_file.subtype_info.split()
                        if temp.isdigit()
                    ][0]

                # get highest and lowest label source (from the tree)
                highest_label_source = annotation.label_source.split("|")[
                    0
                ]  # required for audio name
                lowest_label_source = annotation.label_source.split("|")[
                    -1
                ]  # required for audio name

                # get audio name
                audio_name = (
                    f"{highest_label_source}_"
                    f"{lowest_label_source}_"
                    f"{sea_zone}_"
                    f"{sr}_"
                    f"{byte_sample}_"
                    f"{duration}_"
                    f"{annotation_index}.{extension}"
                )

                # export audio file
                sf.write(
                    os.path.join(
                        output_dir,
                        audio_name,
                    ),
                    audio_array_annotation,
                    sr,
                )

                # relative file path
                file_path = os.path.relpath(file.file_path, datadir)

                # append annotation to annotations_df
                annotations_df.loc[annotation_index - 1] = [
                    audio_name,
                    file_path,
                    annotation.label_source,
                    annotation.label_type,
                    annotation.tmin,
                    annotation.tmax,
                    annotation.fmin,
                    annotation.fmax,
                ]
                logging.info(f"Exported annotation {annotation_index} as {audio_name}")
                annotation_index += 1

        # check metadata, such that it is filled in in metadata.xlsx
        self.meta_check_results()
        self.to_excel(
            os.path.join(output_dir, "metadata.xlsx"), annotations_df=annotations_df
        )

        # export annotations to csv
        annotations_df.to_csv(os.path.join(output_dir, "annotations.csv"), sep=";", decimal = ",") #CHANGED
        logging.info(f"Exported annotations.csv to {output_dir}")
