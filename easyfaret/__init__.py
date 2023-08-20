from __future__ import annotations

__version__ = "0.0.1"
from pathlib import Path
from typing import Literal
from facenet_pytorch import MTCNN

from PIL import Image
from tqdm import tqdm

CKPT_PATH: Path = Path(__file__).parent / "ckpt"

from .dbs.crawler import Crawler
from .dbs.faissdb import FaissDB
from .representation.farl import FaRL


def set_ckpt_path(path: Path):
    global CKPT_PATH
    CKPT_PATH = path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


class EasyFaceRetreival:
    @staticmethod
    def from_queries(
        queries: list[str],
        save_dir: str = "data",
        max_num: int = 30,
    ) -> EasyFaceRetreival:
        crawler = Crawler(save_dir)
        crawler.crawl(queries, max_num=max_num)
        retreival = EasyFaceRetreival()
        retreival.insert(save_dir)
        return retreival

    @staticmethod
    def from_images(
        image_dir: str = "data",
    ) -> EasyFaceRetreival:
        retreival = EasyFaceRetreival()
        retreival.insert(image_dir)
        return retreival

    def __init__(
        self,
        check_face: bool = True,
        db_name: Literal["faiss"] = "faiss",
        face_embidding: Literal["farl"] = "farl",
        face_detector: Literal["mtcnn"] = "mtcnn",
    ):
        # TODO: 다른 모델들 추가
        self._farl = FaRL()
        self._db = FaissDB()
        self._face_detector = MTCNN()
        self._check_face = check_face

    def insert(self, image_dir: str | Path):
        print("DB에 이미지 등록중...")
        is_file = isinstance(image_dir, Path) and image_dir.is_file()
        if is_file:
            images = [image_dir]

        else:
            images = (
                p.resolve()
                for p in Path(image_dir).glob("**/*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            )

            images = list(images)
        assert len(images) > 0, "No images found"

        metadatas = []
        embeddings = []

        model = self._farl

        for img_path in tqdm(images):
            img = Image.open(img_path)
            img_emb = model.get_image_embedding(img)
            metadata = model.get_tags(img)
            name = img_path.parent.name
            metadata["name"] = name
            metadata["file"] = img_path.absolute()
            metadatas.append(metadata)
            embeddings.append(img_emb)

        self._db.insert(embeddings, metadatas)
        print("DB에 이미지 등록 완료")

    def search(self, image: Image.Image, n_results: int = 5) -> dict:
        is_face = self._face_detector(image)
        if is_face is None:
            return {}
        img_emb = self._farl.get_image_embedding(image)
        return self._db.search(img_emb, n_results=n_results)
