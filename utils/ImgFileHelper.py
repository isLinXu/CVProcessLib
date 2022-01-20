import os
import re
from typing import Union
from werkzeug.datastructures import FileStorage
from flask_uploads import UploadSet, IMAGES
'''
pip install flask_uploads
'''


# IMAGES : ('jpg jpe jpeg png gif svg bmp')

# (image set name, allowed extensions)
IMAGE_SET = UploadSet("images", IMAGES)


def save_image(image: FileStorage, folder: str = None, name: str = None) -> str:
    '''
    save image
    '''
    return IMAGE_SET.save(image, folder, name)


def get_path(filename: str = None, folder: str = None) -> str:
    '''
    return image path
    '''
    return IMAGE_SET.path(filename, folder)


def find_image_any_format(filename: str, folder: str) -> Union[str, None]:
    for format in IMAGES:
        image = f'{filename}.{format}'
        image_path = IMAGE_SET.path(filename=image, folder=folder)
        if os.path.isfile(image_path):
            return image_path


def retrieve_filename(file: Union[str, FileStorage]) -> str:
    if isinstance(file, FileStorage):  # check if the input is a FileStorage or just a string
        return file.filename
    return file


def is_file_safe(file: Union[str, FileStorage]) -> bool:
    filename = retrieve_filename(file)
    allowed_format = "|".join(IMAGES)  # jpg|jpe|jpeg|png|gif|svg|bmp
    regex = f"^[a-zA-Z0-9][a-zA-Z0-9_()-\.]*\.({allowed_format})$"
    return re.match(regex, filename) is not None  # return a bool on wether the filename is good


def get_basename(file: Union[str, FileStorage]) -> str:
    filename = retrieve_filename(file)
    return os.path.split(filename)[1]  # some/folder/image.png -> image.png


def get_extension(file: Union[str, FileStorage]) -> str:
    filename = retrieve_filename(file)
    return os.path.splitext(filename)[1]  # some/folder/image.png -> .png