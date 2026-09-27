# 使っている部品のライセンス

## DeepFilterNet（モデルの重み・ONNX 版・前後の計算）

- 出どころ: https://github.com/Rikorose/DeepFilterNet
- ライセンス: **MIT または Apache-2.0**（どちらかを選べる）
- Copyright (c) 2021 Hendrik Schröter
- 使っている箇所:
  - `dfn3/models/dfn3_stateful.onnx` — 公式の `models/DeepFilterNet3_onnx.tar.gz` から
    `tools/make_stateful_onnx.py` で組み替えたもの（重みは変えていない）
  - `dfn3/dfn3-core.js` — `libDF`（Rust）の前後の計算を JavaScript に移植したもの
- 条文: `dfn3/models/DeepFilterNet-LICENSE-MIT`・`dfn3/models/DeepFilterNet-LICENSE-APACHE`

## ONNX Runtime Web 1.30.0

- 出どころ: npm の `onnxruntime-web@1.30.0`（`dist/` の3ファイルをそのまま置いている）
- ライセンス: **MIT**
- Copyright (c) Microsoft Corporation
- 使っている箇所: `dfn3/vendor/ort/`
- 条文: `dfn3/vendor/ort/LICENSE`。実行ファイル（`.wasm`）に含まれる部品の一覧と条文は
  `dfn3/vendor/ort/ThirdPartyNotices.txt`
