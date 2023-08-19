from icrawler.builtin import GoogleImageCrawler


class Crawler:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir