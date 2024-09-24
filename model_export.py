import sys

from transformer_ee.inference.export import export_model_with_normalization


def print_help():
    """
    Print the help message.
    """
    print("Usage: python3 model_export.py [path/to/model/directory]")


# Check if the user requested help or didn't provide a string
if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
    print_help()
else:
    # Capture the command-line argument
    model_dir = sys.argv[1]
    export_model_with_normalization(model_dir)
