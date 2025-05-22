"""
Module for Station class

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
import pandas as pd

# Internal modules
from soundsignature.system import System


class Station:

    def __init__(
        self,
        uid: int,
        name: str,
        campaign: str,
        owner: int,
        institution: int,
        contact: str,
        country_code: str,
        start_date: dt.datetime,
        end_date: dt.datetime,
        zone: str,
        longitude: float,
        latitude: float,
        depth: float,
        systems: list[System],
    ) -> None:
        """
        This class contains information about the station utilised for recording
        underwater noise. A station may have several recording systems (a recorder/hydrophone
        pair with respective parameters gain, sensitivity and depth), which record several audio
        data (name, initial recording time, if annotated, labels, and any comments).

        The required information of a station is documented in the report [D3 Documentation of Standards chosen.pdf]
        which is a deliverable of the CINEA project [CINEA/CD(2022)5010/PP/SI2.899121]:
           > Constructing an open library containing a curated and
           > continuously growing digital catalogue of individual sound
           > signatures from the marine underwater soundscape in
           > shallow seas

        Parameters
        ----------
        name : str
            Name of the station
        campaign : str
            Name of the project or monitoring campaign.
        owner : int
            Name of the data owner [EDMO code].
        institution : int
            Institution which acquired the data [EDMO code].
        contact : str
            Point of contact (institutional email address) of future external queries/who
            submits/holds responsibility for submission.
        country_code : str
            ISO-3166 alpha2 code of the country of the institution.
        start_date : dt.datetime
            Overall campaign collection start date. UTC DateTime in ISO 8601 format:
            YYYY-MM-DDThh:mm[:ss] or YYYY-MM-DD hh:mm[:ss].
        end_date : dt.datetime
            Overall campaign collection end date. UTC DateTime in ISO 8601 format:
            YYYY-MM-DDThh:mm[:ss] or YYYY-MM-DD hh:mm[:ss].
        zone : str
            Geographical region, e.g. zone of the list of European commercial fishing areas,
            such as:
            - 27.4.b (Central North Sea)
            - 37.1.3 (Sardinia)
            or
            MSFD marine sub-regions, such as:
            - Celtic Seas
            - Western Mediterranean Sea
        longitude : float
            Longitude coordinate of station, in WGS84 (EPSG:4326). Notation in decimal degrees [°].
        latitude : float
            Latitude coordinate of station, in WGS84 (EPSG:4326). Notation in decimal degrees [°].
        depth : float
            Level of seabed, in meters below mean sea level [MSL].
        systems : list[System]
            List of soundsignature.Systems associated to this Station.
        """
        self.uid = uid
        self.name = name
        self.campaign = campaign
        self.owner = owner
        self.institution = institution
        self.contact = contact
        self.country_code = country_code
        self.start_date = start_date
        self.end_date = end_date
        self.zone = zone
        self.name = name
        self.longitude = longitude
        self.latitude = latitude
        self.depth = depth
        self.systems = systems

    def __repr__(self):
        return f"Station {self.name}"

    @property
    def in_europe(self):
        """Function to check if current station is in europe"""
        if (30 <= self.latitude <= 75) and (-15 <= self.longitude <= 35):
            return True
        else:
            return False

    @property
    def in_shallow_water(self):
        """Function to check if the water depth is shallow [<100]"""
        if abs(self.depth) < 100:
            return True
        else:
            return False

    @property
    def check_dict(self):
        """Property to return metadata quality of current station"""
        station_meta_dict = {
            "in_europe": self.in_europe,
            "in_shallow_water": self.in_shallow_water,
        }
        for system in self.systems:
            station_meta_dict[system] = system.check_dict
        return station_meta_dict

    def severity_check(self, test, severity_of_tests):
        """If a test failed, print warning or error based on severity."""
        if severity_of_tests[test] == "high":
            print(f"ERROR: station [{self.name}] failed {test} with high severity!")
        elif severity_of_tests[test] == "low":
            print(f"WARNING: station [{self.name}] failed {test} with low severity!")
        else:
            print(
                f"ERROR: station [{self.name}] has incorrect severity: [{severity_of_tests[test]}], "
                "please use 'low' or 'high'."
            )

    @property
    def files(self):
        """Function to get a list of all files within the current station."""
        list_of_files = []

        # Files of a station are all the files of all the underlying systems
        for system in self.systems:
            list_of_files += system.files

        return list_of_files

    @classmethod
    def from_frame(cls, stationID, stations_dataframe, system_list):
        """Function to return the station associated to stationID"""

        # Get particular, single row, according to stationID, and return to series
        station_row = stations_dataframe.loc[
            stations_dataframe["StationID"] == stationID
        ].squeeze()

        # To verify that station row is just a single row / pd.Series
        if not isinstance(station_row, pd.Series):
            ValueError(f"Multiple rows selected by station ID [{stationID}]")

        # Parse all information of series towards initialization of station object
        return cls(
            uid=station_row["StationID"],
            name=station_row["Name"],
            campaign=station_row["Campaign"],
            owner=station_row["Owner"],
            institution=station_row["Institution"],
            contact=station_row["Contact"],
            country_code=station_row["CountryCode"],
            start_date=station_row["StartDate campaign"],
            end_date=station_row["EndDate campaign"],
            zone=station_row["Zone"],
            longitude=station_row["Longitude"],
            latitude=station_row["Latitude"],
            depth=station_row["Bathymetric Depth"],
            systems=system_list,
        )

    def to_dict(self) -> dict:
        """Function to return dictionary of the station"""

        return {
            "Campaign": self.campaign,
            "Owner": self.owner,
            "Institution": self.institution,
            "Contact": self.contact,
            "CountryCode": self.country_code,
            "StartDate campaign": self.start_date,
            "EndDate campaign": self.end_date,
            "Zone": self.zone,
            "Name": self.name,
            "Longitude": self.longitude,
            "Latitude": self.latitude,
            "Bathymetric Depth": self.depth,
        }

    @classmethod
    def from_series(cls, station_row: pd.Series) -> "Station":
        # Parse all information of series towards initialization of station object
        return cls(
            uid=station_row["StationID"],
            name=station_row["Name"],
            campaign=station_row["Campaign"],
            owner=station_row["Owner"],
            institution=station_row["Institution"],
            contact=station_row["Contact"],
            country_code=station_row["CountryCode"],
            start_date=station_row["StartDate campaign"],
            end_date=station_row["EndDate campaign"],
            zone=station_row["Zone"],
            longitude=station_row["Longitude"],
            latitude=station_row["Latitude"],
            depth=station_row["Bathymetric Depth"],
            systems=[],
        )
