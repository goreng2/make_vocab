from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
import os


def parse_corpus_path():
    corpus_paths = []
    for current, dirs, files in os.walk("corpus"):
        for file in files:
            corpus_paths.append(os.path.join(current, file))
    return corpus_paths


def WPM(size, corpora):
    wpm_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    trainer = WordPieceTrainer(
        vocab_size=size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    wpm_tokenizer.train(corpora, trainer)
    wpm_tokenizer.save("vocab/WPM_everyone.json")


def main():
    # corpus_paths = parse_corpus_path()
    corpus_paths = ["corpus/corpus_pp.txt"]
    WPM(32000, corpus_paths)

    # with open(output_json, "r", encoding="utf-8") as f:
    #     body = json.loads(f.read())
    # vocab = body["model"]["vocab"]
    #
    # with open(output, "w", encoding="utf-8") as f:
    #     f.write("\n".join(vocab.keys()))


if __name__ == '__main__':
    main()
