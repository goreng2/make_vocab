# Make Vocabulary with SPM, WPM
create vocab.txt from corpus


## Requirement
```
$ pip install -r requirements.txt
```


## Usage
### SPM
can receive only one corpus file
```
$ python SPM.py --help or -h (for detail)
$ python SPM.py --corpus .../corpus.txt --size 32000 --output .../vocab.txt
```

### WPM
can receive multi corpus files
```
$ pip install tokenizers
$ python WPM.py --help or -h (for detail)
$ python WPM.py --corpus .../corpus.txt .../corpus2.txt --size 32000 --output .../vocab.txt
```

### WPM2
can receive multi corpus files
```
$ pip install tokenizers==0.7.0 (default)
$ python WPM2.py --help or -h (for detail)
$ python WPM2.py --corpus .../corpus.txt .../corpus2.txt --size 32000 --limit_alphabet 6000 --output .../vocab.txt
```


## Reference
### Sentence Piece Model, SPM
- https://github.com/google/sentencepiece/blob/master/python/README.md

### Word Piece Model, WPM
- https://monologg.kr/2020/04/27/wordpiece-vocab/
- https://huggingface.co/docs/tokenizers/python/latest/pipeline.html#all-together-a-bert-tokenizer-from-scratch
