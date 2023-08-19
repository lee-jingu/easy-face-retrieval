
__version__ = '0.0.1'

from pathlib import Path
CKPT_PATH: Path = Path(__file__).parent / 'ckpt'

def set_ckpt_path(path: Path):
    global CKPT_PATH
    CKPT_PATH = path

from .dbs.crawler import Crawler
from .dbs.faissdb import FaissDB
from .detector.facenet import FaceDetector
from .representation.farl import FaRL