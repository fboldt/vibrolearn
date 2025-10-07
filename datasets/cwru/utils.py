def get_code_from_faulty_bearing(faulty_bearing):
    if faulty_bearing == 'Drive End':
        return 'DE'
    elif faulty_bearing == 'Fan End':
        return 'FE'
    else:
        return 'DE'


def get_raw_dir_path():
    return "raw_data/cwru"
