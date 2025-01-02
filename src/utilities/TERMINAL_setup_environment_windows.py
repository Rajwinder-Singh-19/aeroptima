import os
import shutil

"""
NOTE
THIS FILE IS MEANT TO BE EXECUTED FROM THE TERMINAL WITH AEROPTIMA ROOT FOLDER AS THE WORKING DIRECTORY.
THIS FILE ONLY WORKS AS INTENDED FOR WINDOWS 11 OPERATING SYSTEM.

DO NOT USE FOR LINUX.
"""

"""
Sets up a Python 3.13 environment optinally with all the dependencies for aeroptima.
"""

os.chdir("../")  # Now the commands will run in root directory as intended
ENV_PATH = os.getcwd() + "/.venv"
PIP = os.getcwd() + "/.venv/Scripts/pip.exe"
# To install dependencies in the environment


def main():
    valid_choices = ["y", "n"]

    if os.path.isdir(ENV_PATH):
        print("[WARNING] -> A python environment already exists.")
        choice = input(
            "Do you want to clear and generate a new environment? [y/n]: "
        ).lower()
        while choice not in valid_choices:
            print("[ERROR] -> Invalid input. Enter either y or n.")
            choice = input(
                "Do you want to clear and generate a new environment? [y/n]: "
            ).lower()
        if choice == "y":
            print("[INPUT] -> Clearing the python environment...")
            shutil.rmtree(ENV_PATH, ignore_errors=True)
            print("[INFO] -> Environment cleared successfully")
        else:
            print("[INFO] -> Existing python environment still intact. Exiting...")
            choice = input("Open in VS Code? [y/n]: ").lower()
            while choice not in valid_choices:
                print("[ERROR] -> Invalid input. Enter either y or n.")
                choice = input("Open in VS Code? [y/n]: ").lower()
            if choice == "y":
                os.system("code .")
            return
    else:
        print("[INFO] -> Creating python environment...")
        os.system("python3.13 -m venv .venv")
        print("[INFO] -> Python environment created successfully.")

    choice = input("Install Dependencies? [y/n]: ").lower()
    while choice not in valid_choices:
        print("[WARN] -> Invalid input. Enter either y or n.")
        choice = input("Install Dependencies? [y/n]: ").lower()
    if choice == "y":
        print("[INFO] -> Installing dependencies...")
        os.system(PIP + " install -e .")

    choice = input("Open in VS Code? [y/n]: ").lower()
    while choice not in valid_choices:
        print("[ERROR] -> Invalid input. Enter either y or n.")
        choice = input("Open in VS Code? [y/n]: ").lower()
    if choice == "y":
        os.system("code .")


if __name__ == "__main__":
    main()
