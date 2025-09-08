from pathlib import Path
import scipy.io


def load_matlab_file(file_path):
    mat = scipy.io.loadmat(file_path)
    return mat


def get_code_from_faulty_bearing(faulty_bearing):
    if faulty_bearing == 'Drive End':
        return 'DE'
    elif faulty_bearing == 'Fan End':
        return 'FE'
    else:
        return 'DE'


def get_raw_dir_path():
    return "raw_data/cwru"

def get_acquisition(mat, source):
    variable_name = None
    for key in mat.keys():
        if source in key:
            variable_name = key
            break
    if variable_name is not None:
        return mat[variable_name]
    else:
        raise KeyError(f"Variable '{variable_name}' not found in the MATLAB file.")

def load_acquisition(register):
    filename = register['filename']
    file_path =  f"{get_raw_dir_path()}/{filename}"
    mat = load_matlab_file(file_path)
    faulty_bearing = get_code_from_faulty_bearing(register['faulty_bearing'])
    return get_acquisition(mat, faulty_bearing)
