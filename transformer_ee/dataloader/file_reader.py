import io
import lzma
import polars as pl

def get_polars_df_from_xz_file(file_path):
    with open(file_path, "rb") as f:
        xz_data = f.read()
    memory_file = io.BytesIO(xz_data)
    print("Decompressing data from xz file", file_path, "to memory...")
    decompressed_data = lzma.decompress(memory_file.read())
    decompressed_memory_file = io.BytesIO(decompressed_data)
    return pl.read_csv(decompressed_memory_file)


def get_polars_df_from_file(file_path):
    """
    Reads a file and returns a Polars DataFrame.

    Args:
        file_path (str): The path to the file.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the data from the file.

    Raises:
        ValueError: If the file format is not supported.
    """
    if file_path.endswith(".csv"):
        return pl.read_csv(file_path)
    elif file_path.endswith(".xz"):
        return get_polars_df_from_xz_file(file_path)
    else:
        raise ValueError("File format not supported")