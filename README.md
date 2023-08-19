# easy-face-retrieval
FaRl, FAISS, Crawler를 이용한 쉽겍 적용할 수 있는 얼굴 검색 DB 서비스


## Envronment

```
Python 3.8+
```

## Installation

### 본인의 환경에 맞는 [PyTorch](https://pytorch.org/get-started/locally/)를 설치해주세요.

### 아래 설치는 CUDA 12.1을 기준으로 합니다 (CUDA 11.4 이상 권장)

- create environment

```bash
conda create -n easyFaret python=3.8
conda activate faret
```

- install pytorch

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
```

- install faiss-gpu

```bash
conda install -c conda-forge faiss-gpu
```


- install other packages

```bash
pip install -r requirements.txt
```

## References

- [Faiss](https://github.com/facebookresearch/faiss)
- [FaRL](https://github.com/FacePerceiver/FaRL)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- [icrawler](https://github.com/hellock/icrawler)