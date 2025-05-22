"""
Script for extracting metadata from files

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
import logging
import os
import xml.etree.ElementTree as ET

import numpy as np

# External modules
import pandas as pd
import soundfile as sf

# Own modules
import soundsignature.utilities as utl


def read_metadata(xml_path, audio_extension=".wav"):
    """
    Read meta data from a xml-file.
    Designed for xml-file that is included in the .sud-files of SOUNDTRAP instruments

    Parameters
    ----------
        xml_path : str
            path of the XML-file of the corresponding sud-file
        audio_extension : str, default='.wav'
            extension of the audio file (default is '.wav').
            Include the dot, since it will be used to replace '.log.xml'

    Returns
    -------
        system_info : dict
        file_info : dict
        sample_info : list of dicts
        snippet_info : list of dicts
    """

    # read the data inside the xml
    tree = ET.parse(xml_path)

    # pre-allocating of the dictionaries
    system_info = {}
    file_info = {}
    sample_info = {}

    # hydrophone model (without spaces)
    system_info["recorder_number"] = tree.find("EVENT/HARDWARE_ID").text.strip()

    # hydrophone number based on filename, first element of the filename)
    system_info["hydrophone_number"] = os.path.basename(xml_path).split(".")[0]

    # gain setting (high/low)
    # if included
    if tree.findall("EVENT/AUDIO"):
        for record in tree.findall("EVENT/AUDIO"):
            if "Gain" in record.attrib:
                system_info["amplification_gain"] = record.attrib["Gain"]
    else:
        system_info["amplification_gain"] = None

    # Read information with soundfile
    audiofile_path = xml_path.replace(".log.xml", audio_extension)

    try:
        with sf.SoundFile(audiofile_path) as file:

            # system info
            system_info["no_channels"] = file.channels

            # sample info
            file_info["frequency"] = file.samplerate
            sample_info["sample_length"] = file.frames
            file_info["file_duration"] = (
                sample_info["sample_length"] / file_info["frequency"]
            )
            # file info
            file_info["format"] = file.format
            file_info["encoding"] = file.subtype_info
            log = file.extra_info.split()
            file_info["byte_count"] = int(log[log.index("Length") + 2])
            file_info["byte_sample"] = [
                int(temp) for temp in file_info["encoding"].split() if temp.isdigit()
            ][0]
            file_info["byte_rate"] = (
                file_info["byte_count"] / file_info["file_duration"] * 8
            )

            file_info["sample_width"] = file_info["byte_sample"] / 8

    # if unable to read the file
    except Exception as e:
        logging.error(f"Error reading {audiofile_path}: {e}")
        return system_info, file_info, sample_info

    # CODEC information
    # for both the continuous wav-file and the snippets
    for record in tree.findall("CFG"):

        if "FTYPE" in record.attrib:
            if not record.find("SUFFIX") == None:

                # info about the wav-file
                if (record.attrib["FTYPE"] == "wav") & (
                    record.find("SUFFIX").text == " wav "
                ):

                    # ID-number in XML-file corresponding to continuous wav-file
                    wav_ID = record.attrib["ID"]

    # recording information of the wav-files (both continuous wav-file and snippets)
    for record in tree.findall(f'.//{"PROC_EVENT"}'):

        # continuous wav-file
        if record.attrib["ID"] == wav_ID:
            file_info = fill_meta_dict(file_info, record)

    # correct the starttime with offset in microseconds
    file_info = combine_start_meta_dict(file_info)

    return system_info, file_info, sample_info


def fill_meta_dict(meta_dict, record):
    # Loop over all information stored within this xml-object
    # QC BOEJ7: If it is all data from the XML, then this could be parsed easier.
    for wfh in record:

        # sample start time (datetime)
        if "SamplingStartTimeUTC" in wfh.attrib.keys():
            meta_dict["start_time"] = pd.to_datetime(list(wfh.attrib.values())[0])

        # sample end time (datetime)
        if "SamplingStopTimeUTC" in wfh.attrib.keys():
            meta_dict["end_time"] = pd.to_datetime(list(wfh.attrib.values())[0])

        # sample start time offset in us (integer)
        if "SamplingStartTimeSubS" in wfh.attrib.keys():
            meta_dict["start_timestamp_us"] = list(wfh.attrib.values())[0]

    return meta_dict


def combine_start_meta_dict(meta_dict):
    """combine start_timestamp and start time offset in microseconds of the XML-file"""

    # convert start time offset to integer in microseconds (strip us in string)
    start_timestamp_us = int(meta_dict["start_timestamp_us"].replace("us", ""))

    # add the start time offset to the start time
    meta_dict["start_time"] += pd.Timedelta(microseconds=start_timestamp_us)

    # remove the start time offset key
    meta_dict.pop("start_timestamp_us")

    return meta_dict


def metadata_to_csv(data_path, output_dir):
    """Function to make metadata from ALL sud files within a given folder.

    Note: make sure the sud files are extracted into at least .wav and .log.xml files.

    This function saves a metadata excel in output_dir.

    # TODO make this function applicable for file and labels
    """

    # get list of all sud_files in this directory
    sud_files = utl.getFilePaths(data_path, extension=".sud")
    print(f"Number of sud-files found: [{len(sud_files)}]")

    # initialize metadata, with all columns according to the standards document [D3]
    metadata = pd.DataFrame(
        index=range(len(sud_files)),
        columns=[
            "FileID",
            "SystemID",
            "Path",
            "frequency (Hz)",
            "sample length",
            "compressed",
            "sample width",
            "encoding",
            "byte count",
            "byte rate",
            "byte sample",
            "duration (s)",
            "format",
        ],
    )

    sud_counter = 0
    # loop over all sud files
    for sud_file in sud_files:

        basename = os.path.splitext(sud_file)[0]
        # from sud_file to xml file naam
        xml_file = basename + ".log.xml"
        wav_file = basename + ".wav"

        _, file_info, sample_info = read_metadata(xml_file)

        # save important metadata
        metadata.loc[sud_counter, "FileID"] = sud_counter
        # metadata.loc[sud_counter, 'SystemID'] #TODO
        metadata.loc[sud_counter, "Path"] = os.path.relpath(
            wav_file, os.getenv("DATADIR")
        )
        metadata.loc[sud_counter, "frequency (Hz)"] = file_info["frequency"]
        metadata.loc[sud_counter, "sample length"] = sample_info["sample_length"]
        metadata.loc[sud_counter, "compressed"] = "No"
        metadata.loc[sud_counter, "sample width"] = file_info["sample_width"]
        metadata.loc[sud_counter, "encoding"] = file_info["encoding"]
        metadata.loc[sud_counter, "byte count"] = file_info["byte_count"]
        metadata.loc[sud_counter, "byte rate"] = file_info["byte_rate"]
        metadata.loc[sud_counter, "byte sample"] = file_info["byte_sample"]
        metadata.loc[sud_counter, "duration (s)"] = file_info["file_duration"]
        metadata.loc[sud_counter, "format"] = file_info["format"]

        sud_counter += 1

    # set index
    metadata = metadata.set_index("FileID")

    # convert dataframe to csv-file
    metadata_filename = os.path.join(output_dir, "metadata.csv")
    metadata.to_csv(metadata_filename)
    print(f"Metadata exported towards [{metadata_filename}]")
