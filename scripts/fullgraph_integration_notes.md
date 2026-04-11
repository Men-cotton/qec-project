# `fullgraph` を NN-based 各 repo に接続するための改造メモ

更新日: 2026-04-12

## 目的

このメモは、`graph/` にある `fullgraph` JSON と
`scripts/reconstruct_graph_detector_coords.py` を出発点として、

- `NN-based/graphqec-paper`
- `NN-based/QEC_GNN-RNN`
- `NN-based/astra`

の 3 repo に学習データを接続するために必要な改造を、
matching 観点で整理したものである。

## 共通の前提

### `graph/` 側に今ある情報

`graph/readme.md` によると、各 JSON は 1 サンプル分で、

- `fullgraph`
- `fullgraph_node_ids`
- `fullgraph_boundary_node_ids`
- `fullgraph_MWPM_*`

などを持つが、既存 supervised training をそのまま回すための教師ラベルは入っていない。
特に repo 側がそのまま欲しがる

- logical observable label
- per-qubit physical error label

は不足している。

- 参照: `graph/readme.md`

### `reconstruct_graph_detector_coords.py` が与えるもの

このスクリプトは、JSON に detector 座標が無いため、対応する Stim circuit を再生成し、
graph node id を detector id / `(x, y, t)` に戻して CSV と plot を出す。
ただし node id と detector id の対応規則は推論ベースであり、厳密保証ではない。

- 参照: `scripts/reconstruct_graph_detector_coords.py`

### 3 repo 共通のボトルネック

どの repo でも、`fullgraph` をそのまま読むだけでは足りない。最低でも次が要る。

1. `fullgraph` から repo ごとの入力形式へ変換する前処理
2. 欠けている教師ラベルの補完

さらに、3 repo の性格はかなり違う。

- `QEC_GNN-RNN`
  - defect graph を入力にするので、`fullgraph` の edge を最も素直に使える
- `graphqec-paper`
  - 欲しいのは defect graph ではなく detector-event vector + code Tanner graph
- `astra`
  - 欲しいのは static Tanner graph + syndrome node input + per-qubit physical error target であり、`fullgraph` とのズレが最も大きい

---

## 1. `graphqec-paper`

## 現状の入力契約

`graphqec-paper` の NN 系は、任意の graph JSON を直接読む設計ではない。
中心は `QuantumCode` インターフェースで、

- `get_tanner_graph()`
- `get_syndrome_circuit()`
- `get_dem()`
- `get_exp_data()`

を返す code object を前提にしている。

- 参照: `NN-based/graphqec-paper/graphqec/qecc/code.py` / `QuantumCode`

学習データローダは `QuantumCode` から raw detector-event vector を受け取り、

- `encoding_syndromes`
- `cycle_syndromes`
- `readout_syndromes`

に切り分ける。

- 参照: `NN-based/graphqec-paper/graphqec/decoder/nn/dataloader.py`

モデル本体も `TemporalTannerGraph` 上の

- `data_to_check`
- `data_to_logical`

を使っており、入力は defect-defect graph ではなく code Tanner graph である。

- 参照: `NN-based/graphqec-paper/graphqec/qecc/code.py` / `TemporalTannerGraph`, `TannerGraph`
- 参照: `NN-based/graphqec-paper/graphqec/decoder/nn/models.py`

### この repo での意味

`graphqec-paper` が欲しいのは `detector event 列 + code の Tanner graph` であり、
`graph/readme.md` の `fullgraph` が表す `候補辺付き defect graph` ではない。

したがって、この repo に接続する最小案は
`fullgraph を detector-event sample の保存形式として使い、edge 自体は学習に使わない`
である。

## 最小改造案

`fullgraph` を raw detector-event sample へ戻し、モデルは現状のまま使う。

必要な改造は次。

1. `QuantumCode` 実装を 1 つ追加する
   - 例: `NN-based/graphqec-paper/graphqec/qecc/surface_code/local_fullgraph_memory.py`
   - `get_exp_data(num_cycle)` で、`graph/` JSON 群から復元した raw detector-event vector と logical label を返す
   - `get_tanner_graph()` は Stim から構築する
2. `graphqec.qecc.__init__` と code factory に新 code を登録する
3. detector-event 復元前処理を書く
   - `reconstruct_graph_detector_coords.py` の mapping 規則を使って、各 sample を detector-order の二値ベクトルへ戻す
   - `encoding/cycle/readout` の切り分けは現行 dataloader と同じ規約に合わせる
4. logical label を sidecar で与える
   - `graphqec-paper` の評価系は logical observable を前提にしている

### この案の特徴

- 良い点:
  - repo 本体の NN をほぼ触らずに済む
  - `graphqec-paper` の DEM / Tanner-graph 基盤を再利用できる
- 悪い点:
  - `fullgraph` の edge は学習に使われない
  - node-id -> detector-id 対応が推論ベースなので、前処理の健全性確認が要る

## `fullgraph` edge を本当に使う場合

`fullgraph` 辺をそのままモデルに入れたいなら、最小改造では済まない。
必要なのは次。

1. `QuantumCode + raw detector-vector` 前提をやめる
2. `decoder/nn/dataloader.py` を defect-graph loader に差し替える
3. `QECCDecoder._decode()` の raw detector-vector 分割前提を外す
4. `GraphRNNDecoderV5` / `GraphLinearAttnDecoderV2` の `TannerGraph` 依存を外し、defect-defect edge list ベースの encoder に置き換える

これは事実上、`graphqec-paper` の NN decoder を別物に差し替える規模の変更である。

## 推奨

`graphqec-paper` に接続するなら、現実的なのは
`fullgraph` から raw detector-event vector を再構成して既存モデルへ流す
案だけである。

`fullgraph` 辺を主役にしたいなら、この repo は土台としてはあまり適していない。

---

## 2. `QEC_GNN-RNN`

## 現状の入力契約

この repo は `Dataset.generate_batch()` が全入力を作る。流れは次。

1. `stim.Circuit.generated(...)` を作る
2. `compile_detector_sampler()` から `detection_events` と `observable_flips` を取る
3. fired detector を node feature に変換する
4. `knn_graph()` で edge を張る
5. `GRUDecoder` に渡す

- 参照: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.__init__`, `Dataset.sample_syndromes`, `Dataset.get_node_features`, `Dataset.get_edges`, `Dataset.generate_batch`
- 参照: `NN-based/QEC_GNN-RNN/gru_decoder.py` / `GRUDecoder.forward`

モデル側は

- `x`
- `edge_index`
- `edge_attr`
- `batch_labels`
- `label_map`

しか見ていないので、入力 graph を差し替えやすい。

- 参照: `NN-based/QEC_GNN-RNN/gru_decoder.py` / `GRUDecoder.forward`

### この repo での意味

3 つの候補の中では、`fullgraph` を最も自然に受け入れられる。
現在の `knn_graph()` を `fullgraph` に置き換えればよく、model 本体はほぼ触らずに済む。

## 必要な改造

### 1. 入力モードを追加する

`Args` に例えば次を追加する。

- `input_mode: Literal["stim", "fullgraph"]`
- `graph_root`
- `mapping_root`
- `label_path`

現状 `Args` は Stim 生成前提である。

- 参照: `NN-based/QEC_GNN-RNN/args.py`

### 2. `Dataset` を二系統に分ける

現状の `Dataset` は `__init__` で必ず circuit を作る。
ここは次のどちらかに変える必要がある。

- `Dataset` に `stim` / `fullgraph` 分岐を入れる
- `StimDataset` と `FullGraphDataset` に分割する

後者の方が安全である。

### 3. `fullgraph` 用 sample loader を追加する

各 JSON から次を作る。

1. `node_features`
   - `reconstruct_graph_detector_coords.py` の CSV から `(x, y, t)` を取る
   - stabilizer type は現行と同じく座標 parity から再計算する
2. `edge_index`
   - `fullgraph` の `[src, dst, weight]` を PyG 用 edge list に変換する
3. `edge_attr`
   - 保存済み weight をそのまま使うか、単調変換して使う
4. `labels`, `label_map`
   - 最初は `1 sample = 1 graph` でよい
5. `flips`
   - logical label を sidecar から読む

### 4. batch 生成を online sampling から有限 dataset へ変える

現行 `train_model()` は online に `dataset.generate_batch()` を呼ぶ。
`fullgraph` モードでは dataset は有限集合なので、

- `generate_batch()` を事前ロード済み sample から mini-batch を返す形に置き換える
- または `train_model()` / `test_model()` を `DataLoader` ループへ変える

のどちらかが要る。

- 参照: `NN-based/QEC_GNN-RNN/gru_decoder.py` / `GRUDecoder.train_model`, `GRUDecoder.test_model`

### 5. logical label の供給方法を決める

この repo の loss は `observable_flips` を直接学習する。
したがって `graph/` 側に label が無いままでは supervised training はできない。

- 参照: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.sample_syndromes`

## 推奨

3 repo の中では、`QEC_GNN-RNN` が最も素直な接続先である。
必要な改造はあるが、model はそのままで、

- node feature
- edge list
- edge weight
- graph-level label

の供給部分だけを差し替えやすい。

---

## 3. `astra`

## 現状の入力契約

`astra` は `fullgraph` 的な defect graph を直接読む設計ではない。
主経路は次である。

1. `RotatedPlanar2DCode(dist)` を作る
2. `generate_syndrome_error_volume(...)` で `syndrome + physical error` を作る
3. `adapt_trainset(...)` で syndrome node を one-hot 化し、data-qubit node をゼロ埋めして `targets` と対にする
4. `collate(...)` が固定 Tanner graph を batch 連結する
5. `GNNDecoder.forward(node_inputs, src_ids, dst_ids)` が Tanner graph 上で message passing する

- 参照: `NN-based/astra/panq_functions.py` / `surface_code_edges`, `generate_syndrome_error_volume`, `adapt_trainset`, `collate`, `GNNDecoder.forward`
- 参照: `NN-based/astra/gnn_train.py`

重要なのは、local repo が明示的に持っている graph は `surface_code_edges(code)` が作る Tanner graph であり、
`fullgraph` のような defect-defect graph ではない点である。

- 参照: `NN-based/astra/panq_functions.py` / `surface_code_edges`

また、2 段目デコーダは別経路であり、

- `MatchingDecoder`
- `BeliefPropagationOSDDecoder`

を residual syndrome に対して後段で呼んでいる。

- 参照: `NN-based/astra/gnn_test.py`
- 参照: `NN-based/astra/gnn_osd.py` / `osd`, `logical_error_rate_osd`

### この repo での意味

`astra` に接続する場合、`fullgraph` をそのまま GNN 入力 graph にするのは自然ではない。
なぜなら現行 GNN は

- node 種別: `syndrome node + data-qubit node`
- edge: parity-check matrix 由来の Tanner graph
- target: per-qubit 4 値 physical error label

を前提にしているからである。

`fullgraph` が持っている

- defect node
- defect-defect edge
- MWPM 補助情報

とは表現がずれている。

## 最小改造案

`fullgraph` の edge は使わず、
`fullgraph` から syndrome を再構成して既存の Tanner-graph GNN に流す。

必要な改造は次。

1. `fullgraph` から syndrome vector を復元する前処理を書く
   - fired defect node を PanQEC の syndrome 順序へ落とし直す
   - これは `RotatedPlanar2DCode.measure_syndrome(...)` が返す配列順に合わせる必要がある
2. `generate_syndrome_error_volume(...)` の代わりに、外部データ読込 path を追加する
3. `adapt_trainset(...)` に渡せる形へ変換する
   - `inputs`: syndrome node one-hot + data node zero padding
   - `targets`: 現行は `syndrome + per-qubit error label`
4. 不足ラベルを sidecar で与える
   - 最低でも per-qubit の physical error target が必要

### この案の問題点

ここが `QEC_GNN-RNN` や `graphqec-paper` より重い。
`astra` の教師信号は logical bit ではなく、data qubit ごとの 4 値 Pauli label だからである。

`graph/` に logical label が無いだけでなく、`astra` がそのまま欲しい

- `error[:,:d**2] + 2*error[:,d**2:]`

相当の target も無い。

- 参照: `NN-based/astra/panq_functions.py` / `generate_syndrome_error_volume`

したがって、既存学習系をそのまま再利用するには、
元データ生成系から per-qubit error label を sidecar で持ってくる必要がある。

## `fullgraph` edge を本当に使う場合

`fullgraph` を defect graph としてそのまま入れたいなら、かなり大きな改造が要る。

必要なのは次。

1. `surface_code_edges(code)` 由来の static Tanner graph をやめる
2. `collate(...)` を defect-graph batcher に置き換える
3. `GNNDecoder` の node 意味論を変える
   - 現在は syndrome node と data-qubit node を混在させている
   - `fullgraph` では defect node のみになる
4. 出力 head と loss を変える
   - 現在は per-qubit physical error classification
   - `fullgraph` 側に自然なのは logical label か matching-weight 予測

これは `astra` の既存 GNN を表現レベルから作り直すのに近い。

## 2 段目 matching だけを活かす案

`astra` は GNN 単体よりむしろ
`GNN -> fllrx/fllrz -> MatchingDecoder/BP-OSD`
の 2 段構成に特徴がある。

そのため、`fullgraph` を使う別案としては

1. GNN 部は捨てる
2. `fullgraph_MWPM_*` や別モデル出力から qubit-wise weight を作る
3. `MatchingDecoder(..., weights=(fllrx, fllrz))` だけを再利用する

という方向もあり得る。

ただしこれも、`fullgraph` の edge weight をそのまま `fllrx/fllrz` に変換できるわけではない。
`astra` の後段 decoder は defect-edge weight ではなく qubit-wise log-likelihood を受け取るからである。

- 参照: `NN-based/astra/panq_functions.py` / `osd`
- 参照: `NN-based/astra/gnn_osd.py` / `osd`, `init_log_probs_of_decoder`

## 推奨

`astra` は matching-aware ではあるが、`fullgraph` 接続先としては 3 本の中で最も重い。

理由は次。

1. local graph が defect graph ではなく Tanner graph
2. supervised target が logical bit ではなく per-qubit physical error
3. 後段 matching も defect-edge weight ではなく qubit-wise weight を要求する

したがって、`astra` は

- 既存 GNN をそのまま使う接続先

としては優先度が低い。

使うなら、

- Tanner-graph GNN を再利用するのか
- 後段 `MatchingDecoder` だけ再利用するのか

を最初に決めてから入るべきである。

---

## まとめ

### 改造の重さ

1. `QEC_GNN-RNN`
   - 最も現実的
   - `fullgraph` を edge list としてそのまま使いやすい
2. `graphqec-paper`
   - raw detector-event 再構成だけなら可能
   - `fullgraph` 辺を本当に使うなら大改造
3. `astra`
   - matching-aware だが入力契約のズレが大きい
   - 特に per-qubit physical error target が無いのが重い

### repo ごとの向き不向き

- `QEC_GNN-RNN`
  - `fullgraph` の edge / node / weight を最も素直に受けられる
- `graphqec-paper`
  - edge ではなく detector-event vector に戻して使うなら有力
- `astra`
  - `fullgraph` の matching 情報をそのまま活かす先ではなく、Tanner-graph GNN または後段 decoder を切り出して使う先

### 共通して先に決めるべきこと

1. 学習ターゲットを何にするか
   - logical flip
   - per-qubit physical error
   - qubit-wise weight
   - MWPM imitation
2. `reconstruct_graph_detector_coords.py` の node id -> detector id 対応を正式規約として採用するか
3. `fullgraph` の weight を
   - そのまま edge_attr に使うか
   - 別の cost / likelihood に変換して使うか

上の 3 点が固まらないと、どの repo でも実装は途中で詰まる。
