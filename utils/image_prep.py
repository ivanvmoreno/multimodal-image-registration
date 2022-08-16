import multiprocessing as mp
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

# Windows-only
if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(os.environ.get('OPENSLIDE_PATH')):
        import openslide
else:
    import openslide

from PIL import Image


def load_resize_store(
    source: str, out: str, suffix: str, size: tuple, resample = None, max_img_px = 1000000000):
    '''Load an image sample, resize it and store it in an output directory

    :param source: Absolute path to the image samples
    :type source: str
    :param out: Absolute path to the outination folder
    :type out: str
    :param suffix: Image suffix
    :type suffix: str
    :param size: Image size
    :type size: tuple
    :param resample: Image resampling method
    :type resample: str
    '''
    # Set max image size
    Image.MAX_IMAGE_PIXELS = max_img_px

    # Load image sample
    with Image.open(source) as img:
        # Resize image and store it in the output directory
        img = img.resize(size, resample)
        filename, ext = source.split('/')[-1].split('.')
        img.save(f'{out}/{filename}_{suffix}.{ext}')


def load_store_tif_page(
    source: str, out: str, level = 'lowest', suffix: str = 'level'):
    '''Load an TIF sample level and store it in an output directory

    :param source: Absolute path to the image sample
    :type source: str
    :param out: Absolute path to the output directory
    :type out: str
    :param level: TIF level to be loaded. Defaults to the 
        lowest resolution level
    :type level: int
    :param suffix: Resulting filename suffix
    :type suffix: str
    '''
    try:
        # Open TIF file
        img = openslide.OpenSlide(source)

        # Set TIF level to load
        if level == 'lowest':
            level = img.level_count - 1
            
        # Set output filename suffix
        if suffix is None:
            suffix = ''
        else:
            f'_L{level}' if suffix == 'level' else f'_{suffix}'

        # Load TIF page and store it in the output directory
        img = img.read_region((0, 0), level, img.level_dimensions[level])
        filename, ext = source.split('/')[-1].split('.')
        img.save(os.path.join(out, f'{filename}{suffix}.{ext}'))
    except openslide.lowlevel.OpenSlideUnsupportedFormatError as e:
        print(f'Error processing {source} \n {e}')
    

def bulk_process_samples(
    source: str, out: str, samples: list, fun: Callable, *fargs):
    '''Load a list of samples, process them and store them in an
       output path.
    
    :param source: Absolute path to the image samples
    :type source: str
    :param out: Absolute path to the output directory
    :type out: str
    :param samples: List of samples filenames
    :type samples: list
    :param f: Function to be applied to each sample
    :type f: function
    :param fargs: Function positional arguments
    :type fargs: tuple
    '''
    # Check that output directory exists
    Path(out).mkdir(parents=True, exist_ok=True)

    # Start multithreading pool
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(fun, [(f'{source}/{s}', out, *fargs) for s in samples])

    # Close multithreading pool
    pool.close()
