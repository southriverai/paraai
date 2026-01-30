# paraai
Reinforcement learning paragliding policies


## install and activation
So I use gitbash on windows, these command should also work on linux but i am to lazy to test it.

### Getting the main body of dependancies
Installing poetry and create an enviornment
```bash
poetry lock
poetry install
```

### Getting torch
So installing cuda ect is a bit of a rabit hole i dont want to go into, but i use:
```bash
poetry run pip install --upgrade pip
poetry run pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121 \
  torchvision==0.20.1+cu121 \
  torchaudio==2.5.1+cu121
```
### Checking if the GPU is happy
```bash
python script/check_cuda.py
```

## Running the expiriments
I ship with 1 policy included, but if you want to train your own you can use.
```bash
python scripts/train_neural_policy.py
```

This creates all the images and tables used in my speed to fly article
```bash
python scripts/create_all_stf.py
```