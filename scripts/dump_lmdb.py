import lmdb
from sys import argv
from os import makedirs


def dump(path):
    dump_path = f"{path}/dump"
    makedirs(dump_path, exist_ok=True)
    env = lmdb.open(path)
    with env.begin() as txn:
        for key, value in txn.cursor():
            with open(f"{dump_path}/{key.decode()}", "wb") as f:
                f.write(value)


if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python dump_lmdb.py <path>")
    else:
        dump(argv[1])
