# By default, convert all strings to list of floats
# May return empty list if string is empty

def string_to_float_list(string: str) :
    if not string or type(string) != str:
        return []
    return [float(s) for s in string.split(",")]
