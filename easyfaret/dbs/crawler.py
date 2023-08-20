import os

from icrawler.builtin import BingImageCrawler


class Crawler:
    def __init__(self, save_dir: str, check_face: bool = False):
        self.save_dir = save_dir
        self.temp_dir = os.path.join(save_dir, 'temp')

        self.check_face = check_face
        self.filters = dict(
            size='large',
            type='photo',
            people='face',
        )
    
    def crawl(self, queries: list, max_num: int = 100):
        for query in queries:
            root_dir = os.path.join(self.save_dir, query)
            crawler = BingImageCrawler(
                downloader_threads=10,
                storage={'root_dir': root_dir}
            )
            crawler.crawl(keyword=query, filters=self.filters, max_num=max_num)
