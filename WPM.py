from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
import json
import argparse


def train(corpus: list, size: int, output: str) -> None:
    wpm_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    trainer = WordPieceTrainer(
        vocab_size=size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    wpm_tokenizer.train(corpus, trainer)
    wpm_tokenizer.save(output)


def process(output):
    with open(output, "r", encoding="utf-8") as f:
        vocab = json.load(f)["model"]["vocab"]
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab.keys()))


def main(corpus, size, output):
    train(corpus, size, output)
    process(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create WordPieceModel vocabulary")
    parser.add_argument("--corpus", type=str, nargs="+", help="corpus paths")
    parser.add_argument("--size", type=int, default=32000, help="vocab size")
    parser.add_argument("--output", type=str, help="output(vocab) file path/name")
    args = parser.parse_args()

    main(args.corpus, args.size, args.output)