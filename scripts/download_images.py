from pathlib import Path
from argparse import ArgumentParser, Namespace

import repackage
repackage.up()
from easyfaret import Crawler


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='data')
    parser.add_argument('--queries', nargs='+', default=['유재석', '아이유', '박명수'])
    parser.add_argument('--max-num', type=int, default=30)
    return parser.parse_args()

def main():
    args = get_args()
    save_dir: Path = Path(args.save_dir)
    if not save_dir.exists():
        save_dir.mkdir()
    crawler = Crawler(save_dir)
    crawler.crawl(args.queries, args.max_num)


if __name__ == '__main__':
    main()