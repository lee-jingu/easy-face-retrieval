from pathlib import Path
from argparse import ArgumentParser, Namespace

from easyfaret import Crawler


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='data')
    parser.add_argument('--queries', nargs='+', default=['face'])
    parser.add_argument('--max-num', type=int, default=10)
    parser.add_argument('--check', action='store_true')
    return parser.parse_args()

def main(args: Namespace):
    save_dir: Path = Path(args.save_dir)
    if not save_dir.exists():
        save_dir.mkdir()
    crawler = Crawler(save_dir)


if __name__ == '__main__':
    main(get_args())