import argparse
from tqdm import tqdm
import sentencepiece as spm
import os


def train_SPM(corpus: str, size: int) -> None:
    if os.path.exists("sentpiece.vocab"):
        print("[Alert] sentpiece.vocab is already existed")
        return None

    # 공백("_"), "<unk>", "<s>", "</s>"를 제거하기 때문에 미리 4개 추가
    size += 4
    param = """\
    --input={0} \
    --model_prefix=sentpiece \
    --vocab_size={1} \
    --model_type=bpe \
    --user_defined_symbols=[PAD],[UNK],[CLS],[SEP],[MASK] \
    """.format(corpus, size)
    spm.SentencePieceTrainer.Train(param)


def refine_vocab(dest: str) -> None:
    with open('sentpiece.vocab', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    vocab = [line.split("\t")[0] for line in lines]

    for unuse in ["▁", "<unk>", "<s>", "</s>"]:
        vocab.remove(unuse)

    reverse_vocab = []
    for word in tqdm(vocab):
        if word in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
            reverse_vocab.append(word)
        elif word.startswith("▁"):
            reverse_vocab.append(word[1:])
        else:
            reverse_vocab.append("##" + word)

    with open(dest, 'w', encoding='utf-8') as f:
        f.write("\n".join(reverse_vocab))


def main(corpus, size, output):
    train_SPM(corpus, size)
    refine_vocab(output)
    # os.remove("sentpiece.vocab")
    # os.remove("sentpiece.model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make SentencePiece vocab")
    parser.add_argument("--corpus", type=str, help="corpus path")
    parser.add_argument("--size", type=int, default=32000, help="vocab size")
    parser.add_argument("--output", type=str, help="output path")
    args = parser.parse_args()

    main(args.corpus, args.size, args.output)

