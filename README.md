# whisper_realtime

## Overview
Whisperを使ってマイクからの音声をリアルタイムで音声認識するプログラム


## Environment
- Windows 10 Home
  - GPU: RTX2080Ti
- Python3.10
- CUDA 11.4
- CuDNN 8.2.2


## Usage
### 環境構築
```shell
pip install -r requirements.txt
```

### 実行
```shell
python -m whisper_realtime

# コマンドオプション
python -m whisper_realtime -h
usage: __main__.py [-h] [-l] [--mic MIC] [--model MODEL] [--cuda] [--compute_type COMPUTE_TYPE] [--debug]

options:
  -h, --help            show this help message and exit
  -l, --list-devices    デバイスリストを表示して終了する.
  --mic MIC             マイクデバイス名 or マイクID
  --model MODEL         モデルサイズ
  --cuda                GPUを利用する
  --compute_type COMPUTE_TYPE
                        実行形式を指定, [float16(default), int8, int8_float16]
  --debug               debug mode
```


## Author
[T-Sumida](https://twitter.com/sumita_v09)
