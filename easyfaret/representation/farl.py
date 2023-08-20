import os

import torch
import clip
from PIL import Image
from easyfaret import CKPT_PATH

class FaRL:
    def __init__(self):
        path = self.check_and_download()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/16", device="cpu")
        model = model.to(device)
        farl_ckpt = torch.load(path)
        model.load_state_dict(farl_ckpt['state_dict'], strict=False)
        model.eval()

        self.device = device
        self.model = model
        self.preprocess = preprocess

        self.tag_names = dict(
            gender = ['male', 'female'],
            hair_color = ['black hair', 'blond hair', 'brown hair', 'gray hair'],
            skin_color = ['white skin', 'black skin', 'yellow skin'],
            age = ['old', 'young', 'child'],
            glasses = ['with glasses', 'without glasses'],
            costume = ['suit', 'shirt', 't-shirt', 'sweater', 'hoodie', 'jacket', 'dress', 'coat', 'jeans', 'pants', 'shorts', 'skirt'],
            emotion = ['happy', 'sad', 'angry', 'surprised', 'disgusted', 'scared', 'neutral'],
        )

        self.tag_features = {k: model.encode_text(clip.tokenize(v).to(device)) for k, v in self.tag_names.items()}


    def check_and_download(self):
        path = CKPT_PATH / 'farl' / 'FaRL-Base-Patch16-LAIONFace20M-ep16.pth'
        if not path.exists():
            os.makedirs(path.parent, exist_ok=True)
            url = 'https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep16.pth'
            os.system(f'wget {url} -P {path}')
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
        tag_names = self.tag_names

        tags = {}
        with torch.no_grad():
            for k, t_feature in tag_features.items():
                logit_per_image, _ = model(image, t_feature)
                max_idx = torch.argmax(logit_per_image, dim=-1)
                tags[k] = tag_names[k][max_idx]
        
        return tags

    def __call__(self, image: Image.Image) -> dict:
        ret = {
            'embedding': self.get_image_embedding(image),
            'tags': self.get_tags(image),
        }
        return ret