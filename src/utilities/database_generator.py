import os


def to_valid_identifier(filename):
    """
    Converts a filename to a valid Python identifier.
    """
    identifier = filename.replace(" ", "_").replace("-", "_").replace(".", "_")
    if not identifier[0].isalpha():
        identifier = f"_{identifier}"
    return identifier


def generate_typed_dict(folder_name, database_name):
    """
    Generates a TypedDict definition and a dictionary for a given folder's files.
    """
    try:
        filenames = [
            f
            for f in os.listdir(folder_name)
            if os.path.isfile(os.path.join(folder_name, f))
        ]
        identifiers = {to_valid_identifier(f): f for f in filenames}

        # Generate TypedDict definition
        print("from typing import TypedDict\n")
        print("class FileDict(TypedDict):")
        for key in identifiers.keys():
            print(f"    {key}: str")

        # Generate the dictionary
        print(f"\n{database_name}" +": FileDict = {")
        for key, value in identifiers.items():
            print(f'    "{key}": "{value}",')
        print("}")

    except FileNotFoundError:
        print(f"Error: The folder '{folder_name}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    import sys
    generate_typed_dict(os.getcwd() + "/" + str(sys.argv[1]), str(sys.argv[2]))
