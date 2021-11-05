import pickle


def save_file(path, src):
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)


def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def save_pkl(path, src):
    with open(path, "wb") as f:
        pickle.dump(src, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    pass


if __name__ == '__main__':
    main()