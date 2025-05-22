"""
Module with utilities for file handling. No rocket-science, just some snippets

SNIPPED FROM THE PACKAGE `WB_UTILS`


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

import logging
# External modules
import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def check_paths():
    """quick check function to check existence of all paths, provided by the environmental files"""
    if not os.getenv("DATADIR"):
        raise KeyError("Please set DATADIR in env file")
    if not os.path.exists(os.getenv("DATADIR")):
        raise FileNotFoundError(f"Data directory [{os.getenv('DATADIR')}] not found")
    if not os.getenv("EXPORTDIR"):
        raise KeyError("Please set EXPORTDIR in .env file")
    if not os.path.exists(os.getenv("EXPORTDIR")):
        logging.info("Export directory not found, creating it now...")
        os.makedirs(os.getenv("EXPORTDIR"), exist_ok=True)
    if not os.getenv("METADATA"):
        raise KeyError("Please set METADATA in env file")
    metadata_path = os.path.join(os.getenv("DATADIR"), os.getenv("METADATA"))
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Excel file with metadata [{metadata_path}] not found")

    logging.info("All paths are checked and exists")
    return


# %%==============================================================================
def getFilePaths(directory, extension="", pattern=None, onlyRoot=False, caseSensitive=True):
    """
    This function will generate a list of full filenames in a directory tree by
    walking the tree either top-down or bottom-up. For each directory in the
    tree rooted at directory top, it yields a tuple (dir path, dir names, file names).
    These tuples are combined as single string with the full filenames.

    A non-existing directory results in an empty list.


    Parameters
    ----------
    directory : str
        the directory which should be searched (recursively)
    extension : str, default='' (i.e. all files)
        Given an extension, files with only particular extension are returned
    pattern : str, optional
        particular pattern (based on regular expressions, not on fnmatch) on
        which the filenames should be matched
    onlyRoot : bool, default False
        [True]: search only for files in the root directory\n
        [False]: search also files in underlying directories
    caseSensitive : bool, default True
        [True]: filename and extension are matched as given\n
        [False]: extension and filename are matched, based on lower cased format

    Returns
    --------
    list
        a list of full file paths

    Notes
    -----
    os.listdir(directory) returns a list of files WITHOUT absolute path,
    e.g. listDB = [file for file in os.listdir(datadir) if file.endswith('.sqlite')]
    """

    filepaths = []  # List which will store all the full file paths

    # Walk the tree when all sub folders also should be searched
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Only append when the extension matches the desired extension
            # v1.9: and the filename contains PATTERN

            # v1.8: or append if caseSensitivity is not needed
            if (pattern is None) or (re.search(pattern, filename) is not None):
                # In this case: the pattern is found, or there is no pattern given
                if filename.endswith(extension) or (
                    not caseSensitive and filename.lower().endswith(extension.lower())
                ):
                    # Join the two strings in order to form the full filepath
                    filepath = os.path.join(root, filename)
                    filepaths.append(filepath)  # Add it to the list of filePaths

        if onlyRoot:
            # If only the root-folder is needed: then the outer-for-loop should
            # be finished after one iteration. Break the loop, to return the list
            break

    return filepaths  # The list of full file paths


def plotDensityChart(timestamps, vmax=None):
    """
    plotting part of sample-density plot

    """
    # Defining colormap
    mycmap = mpl.cm.GnBu  # pylint: disable=no-member
    mycmap.set_under("white")

    if vmax == None:
        vmax = np.nanmax(timestamps)

    # Initialize a figure, with plot
    fig, ax = plt.subplots()
    im = ax.imshow(timestamps.T, cmap=mycmap, vmin=0, vmax=vmax, aspect="auto")

    # Putting a (very thin) grid in the plot
    ax.grid(alpha=0.25, linewidth=0.5)

    # Each bin-element should contain a tick, but not all ticks should be labeled
    # e.g. (only each week on a sunday, or only first month) PROJECT SPECIFIC!!
    ax.set_xticks(np.arange(0, len(timestamps.index)), minor=True)
    ax.set_xticks(np.arange(0, len(timestamps.index), 12), minor=False)
    datelabels = [
        "{:04d}-{:02d}".format(lab.year, lab.month) if (lab.month == 1) else ""
        for lab in timestamps.index.tolist()
    ]
    ax.set_xticklabels(datelabels, rotation=90, minor=True)
    ax.set_xticklabels([], minor=False)

    # PROJECT SPECIFIC: UNDESIRABLE
    step = 1
    ax.set_yticks(np.arange(0, len(timestamps.columns), step))
    ax.set_yticklabels(timestamps.columns[::step])

    # Creating a margin for top and bottom ticks

    # Doesn't work:
    # ax.set_ylim(bottom=-0.5, top=len(timestamps.columns) + 0.5)
    ax.set_ylim(bottom=len(timestamps.columns) - 0.5, top=-0.5)

    # Add a (smaller than default) colorbar
    fig.colorbar(im, aspect=40)  # PRJ SPC!, ticks=[0, 5, 10, 15, 20, 24])

    return fig, ax
