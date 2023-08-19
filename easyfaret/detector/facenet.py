from facenet_pytorch import MTCNN


class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()