
# Предварительная подготовка


Поставить нужный pytorch (На сервере сейчас образы с CUDA8.0)
```bash
pip install torch==1.0.1 -f https://download.pytorch.org/whl/cu80/stable
```

Поставить torchaudio
```bash
sudo apt-get install sox libsox-dev libsox-fmt-all
pip install git+https://github.com/pytorch/audio
```

# Данные

Нужно скачать 

```bash
ubuntu@ubuntugpu:/data/kaggle-freesound-2019$ ls
total 23G
-rw-rw-r-- 1 ubuntu 191K May 14 13:51 sample_submission.csv
-rw-rw-r-- 1 ubuntu 667M May 14 13:51 test.zip
-rw-rw-r-- 1 ubuntu 140K May 14 13:51 train_curated.csv
-rw-rw-r-- 1 ubuntu 2.3G May 14 13:51 train_curated.zip
-rw-rw-r-- 1 ubuntu 572K May 14 13:50 train_noisy.csv
-rw-rw-r-- 1 ubuntu  21G May 14 13:50 train_noisy.zip
```
и распаковать

```bash
unzip train_curated.zip -d ./train_curated
unzip train_noisy.zip -d ./train_noisy 
```


# Запуск тренировки

```bash
# синхронизовать код
# зайти в нужную папку
python main.py --outpath ./runs/
```

# Посмотреть графики

```bash
# запустить тмукс
# зайти в папку с запусками и запустить TB
CUDA_VISIBLE_DEVICES= tensorboard --logdir=./
```

```bash
# На своей машине запустить
ssh -L 9009:localhost:6006 ubuntu@your_server_ip
# зайти в браузере на http://localhost:9009 
```


# Посылка

- создать приватный кернел с кодом из kernel_infer.py
- залить лучший чекпоинт `best.pth` в качестве датасета
- проверить пути, запустить, дождаться просчета и на вкладке Outputs засабмитить ответы