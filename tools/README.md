# tools

## make_stateful_onnx.py — モデルの作り方

`dfn3/models/dfn3_stateful.onnx` は、公式の DeepFilterNet3 の ONNX 版を組み替えたもの。

**なぜ組み替えるか。** 公式の ONNX 版は「録音全体を一度に処理する」前提で作られていて、
呼ぶたびに内部の記憶（GRU の状態）がゼロから始まる。長い録音を区切って処理すると、
区切りごとに記憶がリセットされ、結果が一度に処理した場合からずれていく
（実測で、区切ってから12秒たってもずれが消えなかった）。
一度に処理すればずれないが、長い録音ではメモリが足りなくなる。

そこで、記憶を外から渡せるようにグラフを組み替えた。

1. 時間方向の詰め物（Pad）を「直前のフレームを受け取る入力」に置き換える
2. 5つの GRU の記憶を、入力と出力として外に出す
3. 3つのファイル（enc / erb_dec / df_dec）を1つにまとめる

**重みは1バイトも変えていない**（学習し直しも、PyTorch からの書き出し直しもしていない）。

### 作り直す手順

```sh
curl -LO https://raw.githubusercontent.com/Rikorose/DeepFilterNet/main/models/DeepFilterNet3_onnx.tar.gz
mkdir onnx && tar -xzf DeepFilterNet3_onnx.tar.gz -C onnx
pip install onnx numpy
python make_stateful_onnx.py onnx/tmp/export dfn3_stateful.onnx
sha256sum dfn3_stateful.onnx
```

2026-09-27 に作り直して、`dfn3/models/dfn3_stateful.onnx` と一致することを確かめた。

| ファイル | sha256 |
|---|---|
| `DeepFilterNet3_onnx.tar.gz`（入力） | `c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616` |
| `dfn3_stateful.onnx`（出力、onnx 1.23.0 で作成） | `442c06aaf6aab2849ad44e203239abe3802da533eba4a129db90f65c3c485a6c` |
