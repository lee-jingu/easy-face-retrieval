import os

import torch
import clip
import wget
from PIL import Image
from easyfaret import CKPT_PATH


class FaRL:
    def __init__(self):
        path = self.check_and_download()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("clip loading...")
        model, preprocess = clip.load(
            "ViT-B/16", device="cpu", download_root=CKPT_PATH / "clip"
        )
        print("clip loaded")
        model = model.to(device)
        farl_ckpt = torch.load(path)
        model.load_state_dict(farl_ckpt["state_dict"], strict=False)
        model.eval()

        self.device = device
        self.model = model
        self.preprocess = preprocess

        # TODO: Config file로 관리
        self.tag_names = dict(
            gender=["male", "female"],
            hair_color=[
                "black hair",
                "blond hair",
                "brown hair",
                "gray hair",
                "white hair",
            ],
            emotion=[
                "happy",
                "sad",
                "angry",
                "surprised",
                "disgusted",
                "scared",
                "neutral",
            ],
        )
        self.tag_names_ko = dict(
            gender=["남자", "여자"],
            hair_color=["검정 머리", "금발", "갈색 머리", "회색머리", "흰머리"],
            emotion=["행복", "슬픔", "분노", "놀람", "역겨움", "두려움", "중립"],
        )

        tag_features = {}
        with torch.no_grad():
            for k, v in self.tag_names.items():
                tag_features[k] = clip.tokenize(v).to(self.device)
        self.tag_features = tag_features

    def check_and_download(self):
        path = CKPT_PATH / "farl" / "FaRL-Base-Patch16-LAIONFace20M-ep16.pth"
        print("얼굴 임베딩 모델 loading...")
        if not path.exists():
            os.makedirs(path.parent, exist_ok=True)
            url = "https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep16.pth"
            wget.download(url, str(path))
        print("얼굴 임베딩 모델 loaded")
        return path

    def get_image_embedding(self, image: Image.Image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features.detach().cpu().numpy().squeeze()

    def get_tags(self, image: Image.Image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        model = self.model
        tag_features = self.tag_features
        tag_names = self.tag_names_ko

        tags = {}
        with torch.no_grad():
            for k, t_feature in tag_features.items():
                logit_per_image, _ = model(image, t_feature)
                max_idx = torch.argmax(logit_per_image, dim=-1)
                tags[k] = tag_names[k][max_idx]

        return tags
