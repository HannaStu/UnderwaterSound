"""
Module for Annotation class

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

import datetime as dt
import logging
import os

import pandas as pd


class Annotation:
    """This class contains all information about an annotation of an audio sample.

    Parameters
    ----------
    parent_file : File
        The File that contains current sample.
    institution : str
        Name of institution that performs the labelling.
    process : str
        Process of labelling, i.e. manual or automatic (name of algorithm).
    method : str
        Link  (URL or DOI) to used methodology.
    date : dt.datetime
        Date of annotation.
    status : str
        not annotated / in progress / done.
    annotator_id : int
        Identification number of the annotator.
    tmin : float
        Initial time of sound event [s].
    tmax : float
        Final time of sound event [s].
    fmin : float
        Minimum frequency of event [Hz].
    fmax : float
        Maximum frequency of event [Hz].
    label_source : str
        Source of sound event, i.e. biologic, anthropogenic or unknown.
    label_type : str
        Signal nature of sound event, e.g. continuous, pulse, etc.
    comments : str
        Free remarks in English.
    """

    def __init__(
        self,
        parent_file: None,
        institution: str,
        process: str,
        method: str,
        date: dt.datetime,
        status: str,
        annotator: int,
        tmin: float,
        tmax: float,
        fmin: float,
        fmax: float,
        label_source: str,
        label_type: str,
        reference: str = "",
        comments: str = "",
    ) -> None:

        # Properties of labeling process
        self.parent_file = parent_file
        self.institution = institution
        self.process = process
        self.method = method
        self.date = date
        self.status = status

        # Properties of label itself
        self.annotator = annotator
        self.reference = reference
        self.tmin = tmin
        self.tmax = tmax
        self.fmin = fmin
        self.fmax = fmax
        self.label_source = label_source
        self.label_type = label_type
        self.comments = comments

    def __repr__(self) -> str:
        return (
            f"Annotation with source {self.label_source.split('|')[-1]} and type {self.label_type}"
        )

    def make_annotation_table(self) -> pd.DataFrame:
        """Generate an annotation table describing a sound event"""

        annotation_table = {
            "annotator": [self.annotator],
            "reference": [self.reference],
            "tmin": [self.tmin],
            "tmax": [self.tmax],
            "fmin": [self.fmin],
            "fmax": [self.fmax],
            "label_source": [self.label_source],
            "label_type": [self.label_type],
            "comments": [self.comments],
        }

        annotation_table = pd.DataFrame(annotation_table)

        return annotation_table

    def to_dict(self) -> dict:
        """Function to return dictionary of the label"""

        return {
            "parent_file": self.parent_file,
            "institution": self.institution,
            "process": self.process,
            "method": self.method,
            "date": self.date,
            "status": self.status,
            "annotator": self.annotator,
            "reference": self.reference,
            "tmin": self.tmin,
            "tmax": self.tmax,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "label_source": self.label_source,
            "label_type": self.label_type,
            "comments": self.comments,
        }

    def sanity_check_label(self) -> bool:
        """Check whether the tmin, tmax, fmin and fmax make sense."""

        # Check if tmin < tmax and fmin < fmax, if so return True
        if self.tmax > self.tmin:
            if self.fmax > self.fmin:
                logging.info("Label is valid!")
                return True
            # if fmin > fmax, return False
            elif self.fmax < self.fmin:
                logging.warning(
                    f"Label with label_source {self.label_source} and "
                    f"label_type {self.label_type} is invalid!"
                )
                logging.warning("Warning: fmin > fmax!")
                return False

        # if tmin > tmax, return False
        elif self.tmax < self.tmin:
            if self.fmax > self.fmin:
                logging.warning(
                    f"Label with label_source {self.label_source} and "
                    f"label_type {self.label_type} is invalid!"
                )
                logging.warning("Warning: tmin > tmax!")
                return False
            elif self.fmax < self.fmin:
                logging.warning(
                    f"Label with label_source {self.label_source} and "
                    f"label_type {self.label_type} is invalid!"
                )
                logging.warning("Warning: tmin > tmax, fmin > fmax!")
                return False

    def has_valid_source_label(self) -> bool:
        """
        Check whether the label source is following the standards.
        """
        # check if label source is in list of valid sources
        if self.label_source.split("|")[-1] in LIST_VALID_SOURCES:
            return True

        else:
            logging.warning(
                f"Label with label_source {self.label_source} and "
                f"label_type {self.label_type} is invalid!"
            )
            return False

    def has_valid_type_label(self) -> bool:
        """
        Check whether the label type is following the standards.
        """

        # check if label type is in list of valid types
        if self.label_type in LIST_VALID_TYPES:
            return True
        else:
            logging.warning(
                f"Label with label_source {self.label_source} and "
                f"label_type {self.label_type} is invalid!"
            )
            return False

    @classmethod
    def from_row(cls, file, process_dict, label_row) -> "Annotation":

        # Parse all information of dataframe towards initialization of system object
        if "Comments" in label_row:
            comments = label_row["Comments"]
        else:
            comments = None

        # based on LABEL_SOURCE_TREE, check label source and give all parents from label source
        # to check if label source is valid
        label_source = label_row["Source Label"]
        label_type = label_row["Type Label"]
        expanded_label_source = expand_labels(LABEL_SOURCE_TREE, label_source)
        expanded_label_type = expand_labels(LABEL_TYPE_TREE, label_type)

        return cls(
            # Properties of labeling process
            parent_file=file,
            institution=process_dict["institution"],
            process=process_dict["process"],
            method=process_dict["method"],
            date=process_dict["date"],
            status=process_dict["status"],
            # Properties of label itself
            annotator=process_dict["annotator"],
            reference=label_row["Audio Name"],
            tmin=label_row["Time min (s)"],
            tmax=label_row["Time max (s)"],
            fmin=label_row["Frequency min (Hz)"],
            fmax=label_row["Frequency max (Hz)"],
            label_source=expanded_label_source,
            label_type=expanded_label_type,
            comments=comments,
        )


# The structure of the labels, (including hierarchy).
# It serves as a global variable.
LABEL_SOURCE_TREE = {
    "Biological": {
        "Fishes": {},
        "Benthos": {},
        "Mammals": {
            "Mysticetes": {
                "FinWhale": {},
                "BlueWhale": {},
                "HumpbackWhale": {},
                "MinkeWhale": {},
                "RightWhale": {},
            },
            "Odontocetes": {
                "Delphinids": {
                    "Tursiops": {},
                    "Stenella": {},
                },
                "SpermWhale": {},
                "BeakedWhale": {},
                "PilotWhale": {},
                "KillerWhale": {},
                "Porpoise": {},
                "Beluga": {},
                "Narwhal": {},
            },
            "Pinnipeds": {"Seals": {}, "Otarias": {}, "Walrus": {}},
        },
    },
    "Anthropogenic": {
        "Ship": {
            "MilitaryVessel": {},
            "CargoVessel": {},
            "LeisureShip": {},
            "FishingBoat": {},
            "SmallBoat": {},
        },
        "Sonar": {
            "MilitarySonar": {},
            "CivilianSonar": {},
            "EchoSounder": {
                "SingleBeam": {},
                "MultiBeam": {},
                "Sediment": {},
            },
        },
        "Pinger": {},
        "ActivitySeabed": {
            "LayingPipesCables": {},
            "DredgingMining": {},
            "Drilling": {},
        },
        "PumpingOilGaz": {},
        "MarineRenewableEnergyProduction": {
            "WindmillTurbine": {
                "FixedFoundation": {},
                "Floating": {},
            },
            "TidalTurbine": {},
            "WaveEnergyConvertor": {},
        },
        "PileDriving": {
            "Vibratory": {},
            "Hammer": {},
        },
        "ActiveElectricPowerCables": {},
        "MooringNoise": {"Chains": {}},
        "Explosives": {
            "ShipShockTrial": {},
            "MilitaryActivity": {},
            "SealBombs": {},
            "RockBlasting": {},
        },
        "Airguns": {
            "AirgunEcho": {},
        },
    },
    "Geological": {
        "Weather": {"Wind": {}, "Rainfall": {}, "Thunder": {}},
        "Waves": {},
        "SeabedActivity": {
            "Earthquake": {},
            "HydrothermalVents": {},
            "SubmarineVolcanicEruption": {},
        },
        "IceCracking": {},
        "IcebergCollision": {},
    },
    "Undefined": {},
    "BackgroundNoise": {},
}

LABEL_TYPE_TREE = {
    "Transient": {
        "FrequencyModulation": {},
        "PulseSet": {
            "PulseTrain": {},
        },
        "Pulse": {},
    },
    "Stationary": {},
    "Undefined": {"Saturation": {}},
}


def get_all_keys(mydict: dict):
    """
    A helper function to get ALL the keys in a nested dictionary, recursively.
    It returns a generator. Convert to list or set when needed.
    """
    for key in mydict.keys():
        # first: the key itself (uplevel)
        yield key
        # second: call same function for the content (also dict) below current level
        yield from get_all_keys(mydict[key])


def find_parents(tree, label) -> list:
    """
    A helper function to get all parent keys of a nested dictionary, recursively.
    It returns a list. (appending lists is done with '+')
    """

    for parent, child_dict in tree.items():
        # Either the label is present on highest level of view ...
        if parent == label:
            return [parent]

        # ... or we should search in deeper levels of the tree
        childs = find_parents(child_dict, label)
        if childs:
            # If it found anything, then add it, and return.
            return [parent] + childs

    # If everything is passed, and nothing found, return an empty list.
    return []


def expand_labels(tree, deep_label):
    """helper function to generate an expanded label representation"""
    label_hierarchy = find_parents(tree, deep_label)
    return "|".join(label_hierarchy)


LIST_VALID_SOURCES = list(get_all_keys(LABEL_SOURCE_TREE))
LIST_VALID_TYPES = list(get_all_keys(LABEL_TYPE_TREE))

ANNOTATION_FILES = {
    "borssele-rws-nl": [
        os.path.join("borssele-rws-nl", "annotation", "borssele_labels-SHOM.xlsx"),
        os.path.join("borssele-rws-nl", "annotation", "Waterproof labels - standardized.xlsx"),
    ],
    "jfb-it": [os.path.join("jfb-it", "annotation", "JFB-IT-labels-updated.xlsx")],
    "jomopans-rws-nl": [
        os.path.join("jomopans-rws-nl", "annotation", "jomopans_annotation - standardized.xlsx")
    ],
    "univigo-sp": [os.path.join("univigo-sp", "annotation", "univigo-labels-updated.xlsx")],
    "waddensea-rws-nl": [
        os.path.join("waddensea-rws-nl", "annotation", "labels-Waddensea-SHOM.xlsx")
    ],
    "waveglider-szn-it": [
        os.path.join("waveglider-szn-it", "annotation", "SZN_annotation - standardized.xlsx")
    ],
    "deepship": [os.path.join("deepship", "annotation", "deepship - standardized.xlsx")],
    "safewave": [os.path.join("safewave", "annotation", "safewave - standardized.xlsx")],
    "mambo-fr": [os.path.join("mambo-fr", "annotation", "MAMBO_FR - standardized.xlsx")],
    "wavec": [os.path.join("wavec", "annotation", "wavec - standardized.xlsx")],
    "posa": [os.path.join("posa", "annotation", "posa - standardized.xlsx")],
    "mambo09": [os.path.join("mambo09", "annotation", "mambo09 - standardized.xlsx")],
    "borssele-oper-rws-nl": [
        os.path.join("borssele-oper-rws-nl", "annotation", "borssele-oper - standardized.xlsx")
    ],
}
