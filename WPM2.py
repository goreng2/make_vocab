import argparse
import os
from tokenizers import BertWordPieceTokenizer  # pip install tokenizers==0.7.0


def train(corpus: list, size: int, limit: int, output: str) -> None:
    tokenizer = BertWordPieceTokenizer(
        vocab_file=None,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,  # Must be False if cased model
        lowercase=False,
        wordpieces_prefix="##"
    )
    tokenizer.train(
        files=corpus,
        limit_alphabet=limit,
        vocab_size=size
    )
    # tokenizer.save("./", "ch-{}-wpm-{}".format(args.limit_alphabet, args.vocab_size))
    path, filename = os.path.split(output)
    tokenizer.save(path, filename)


def main(corpus, size, limit, output):
    train(corpus, size, limit, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create WordPieceModel vocabulary")
    parser.add_argument("--corpus", type=str, nargs="+", help="corpus paths")
    parser.add_argument("--size", type=int, default=32000, help="vocab size")
    parser.add_argument("--limit_alphabet", type=int, default=6000, help="num of only one character")
    parser.add_argument("--output", type=str, help="output(vocab) file path/name")
    args = parser.parse_args()

    main(args.corpus, args.size, args.limit_alphabet, args.output)