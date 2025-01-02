import os

"""
NOTE
THIS FILE IS MEANT TO BE EXECUTED FROM THE TERMINAL WITH AEROPTIMA ROOT FOLDER AS THE WORKING DIRECTORY.

PASS THE FOLDERNAME AS THE SYSTEM ARGUMENT IN THE TERMINAL.
"""

"""
Creates a database dictionary from the files in a folder.
"""


def __to_valid_identifier(filename: str) -> str:
    """
    Converts a filename to a valid Python identifier.

    PARAMETERS:

        `filename` -> .dat filename of the file. Type(str)

    RETURNS:

        `identifier` -> Converted filename which can be used as an identifier in Python. Type(str)

    """
    identifier = filename.replace(" ", "_").replace("-", "_").replace(".", "_")
    if not identifier[0].isalpha():
        identifier = f"_{identifier}"
    return identifier


def __generate_typed_dict(folder_name: str, database_name: str) -> None:
    """
    Generates a TypedDict definition and a dictionary for a given folder's files.

    PARAMETERS:

        `folder_name` -> The folder location which is to be converted into a database dictionary. Type(str)

        `database_name` -> Name of the database dictionary. Type(str)

    """
    try:
        filenames = [
            f
            for f in os.listdir(folder_name)
            if os.path.isfile(os.path.join(folder_name, f))
        ]
        identifiers = {__to_valid_identifier(f): f for f in filenames}

        # Generate TypedDict definition
        print("from typing import TypedDict\n")
        print("class FileDict(TypedDict):")
        for key in identifiers.keys():
            print(f"    {key}: str")

        # Generate the dictionary
        print(f"\n{database_name}" + ": FileDict = {")
        for key, value in identifiers.items():
            print(f'    "{key}": "{value}",')
        print("}")

    except FileNotFoundError:
        print(f"Error: The folder '{folder_name}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    import sys

    __generate_typed_dict(os.getcwd() + "/" + str(sys.argv[1]), str(sys.argv[1]))
