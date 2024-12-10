# Password AutoEncoders


## Using

```bash
pipenv install
```
### simple inference
```bash
pipenv run python3 sample.py --pii=/path/to/personal_ident_info.txt --stdout
```

### see more
```bash
usage: sample.py [-h] [--method M] [--wordlist_first FILE] [--log_file FILE] [--load_model FILE] [--repo_id DIR] [--local]
                 [--min_len N] [--max_len N] [--batch_size N] [--similar_sample_n N] [--similar_std STD] [--pii FILE|DIR] [--stdout]
                 [--save_dir DIR]
```

## Базовая архитектура автокодировщика

<img src="img/ae.png" alt="ae" style="zoom:50%;" />





## Базовая архитектура состязательного автокодировщика

<img src="img/aae.png" alt="ae" style="zoom:50%;" />
