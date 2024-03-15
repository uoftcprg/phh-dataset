from collections import Counter
from glob import glob
from pathlib import Path
from sys import argv

from pokerkit import HandHistory


def main():
    for argument in argv[1:]:
        pathnames = glob(argument, recursive=True)
        counts = Counter()

        for pathname in pathnames:
            path = Path(pathname)

            with open(path, 'rb') as file:
                hh = HandHistory.load(file)

            counts[hh.variant] += 1

        print(argument)
        print()

        for key, value in counts.items():
            print(f'{key}: {value}')

        print()
        print()


if __name__ == '__main__':
    main()
