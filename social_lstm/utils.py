import os

def check_path(path):
    '''
    FUNC: convert path (can be realtive path or using ~) to absolute path and check
          the path exists.
    '''
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise NameError('Not such path as {}'.format(path))

    return path

def check_dir(dir_path):
    '''
    FUNC: check directory path, if no such directory, then create one
    '''
    dir_path = os.path.abspath(os.path.expanduser(dir_path))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        if os.path.isdir(dir_path):
            return dir_path
        else:
            raise NameError('{} is not a directory'.format(dir_path))
    
    return dir_path
   
