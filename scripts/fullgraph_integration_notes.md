# `fullgraph` を NN-based 各 repo に接続するための改造メモ

更新日: 2026-04-09

## 目的

このメモは、`graph/` にある `fullgraph` JSON と
`scripts/reconstruct_graph_detector_coords.py` を出発点として、

- `NN-based/graphqec-paper`
- `NN-based/QEC_GNN-RNN`
- `NN-based/neural_network_decoder`

の 3 repo に学習データを接続するために必要な改造を、
実際のコード構造に基づいて整理したものである。

## 共通の前提

### `graph/` 側に今ある情報

`graph/readme.md` によると、各 JSON は 1 サンプル分で、`fullgraph`、`fullgraph_node_ids`、
`fullgraph_boundary_node_ids`、`fullgraph_MWPM_*` などは入っているが、学習用の logical label は
入っていない。特に NN 学習候補として README が明示しているのは MWPM の重みや代表 matching であり、
logical observable は書かれていない。

- 参照: `graph/readme.md:12-37`
- 参照: `graph/readme.md:94-107`

### `reconstruct_graph_detector_coords.py` が与えるもの

このスクリプトは、JSON に detector 座標が無いため、対応する Stim circuit を再生成し、
graph node id を detector id / `(x, y, t)` に戻して CSV と plot を吐く。
ただし node id と detector id の対応規則は「推論」であり、厳密保証ではない。

- 参照: `scripts/reconstruct_graph_detector_coords.py:3-15`
- 参照: `scripts/reconstruct_graph_detector_coords.py:181-213`
- 参照: `scripts/reconstruct_graph_detector_coords.py:237-260`

### 3 repo 共通のボトルネック

どの repo でも、`fullgraph` をそのまま読むだけでは足りない。最低でも次が要る。

1. `fullgraph` から repo ごとの入力形式へ変換する前処理
2. 学習ターゲットの追加

特に `graph/` には logical label が無いので、現状の supervised training をそのまま回すなら、
別ファイルで label を付与する必要がある。label が無いまま使えるのは、MWPM imitation へ
タスクを変える場合だけだが、その場合は各 repo の loss / head も変える必要がある。

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

- 参照: `NN-based/graphqec-paper/graphqec/qecc/code.py:120-146`

学習データローダは `QuantumCode` から raw detector-event vector を受け取り、
`encoding_syndromes`, `cycle_syndromes`, `readout_syndromes` に切り分ける。

- 参照: `NN-based/graphqec-paper/graphqec/decoder/nn/dataloader.py:19-77`
- 参照: `NN-based/graphqec-paper/graphqec/decoder/nn/dataloader.py:92-125`
- 参照: `NN-based/graphqec-paper/graphqec/decoder/nn/dataloader.py:148-188`

モデル本体も `TemporalTannerGraph` 上の

- `data_to_check`
- `data_to_logical`

を使っており、入力は defect-defect graph ではなく code Tanner graph である。

- 参照: `NN-based/graphqec-paper/graphqec/decoder/nn/models.py:12-30`
- 参照: `NN-based/graphqec-paper/graphqec/decoder/nn/models.py:42-58`
- 参照: `NN-based/graphqec-paper/graphqec/decoder/nn/models.py:221-255`

実装例の `SycamoreSurfaceCode` も、`get_exp_data()` は detector-event 行列を返し、
`_get_tanner_graph()` は circuit 幾何から Tanner graph を作る。

- 参照: `NN-based/graphqec-paper/graphqec/qecc/surface_code/google_block_memory.py:148-162`
- 参照: `NN-based/graphqec-paper/graphqec/qecc/surface_code/google_block_memory.py:164-220`

### この repo での意味

`graphqec-paper` が欲しいのは「detector event 列 + code の Tanner graph」であり、
`graph/readme.md` の `fullgraph` が表す「候補辺付き defect graph」ではない。
したがって、`fullgraph` を忠実に使うなら model 側までかなり深く変える必要がある。

## 最小改造案

`fullgraph` を「raw detector-event sample の保存形式」と割り切り、
モデルは現状のまま使う。

必要な改造は次。

1. `QuantumCode` 実装を 1 つ追加する
   - 例えば `NN-based/graphqec-paper/graphqec/qecc/surface_code/local_fullgraph_memory.py`
   - `get_exp_data(num_cycle)` で、`graph/` JSON 群から復元した raw detector-event vector と logical label を返す
   - `get_tanner_graph()` は、Stim `surface_code:rotated_memory_z` の幾何から構成する
2. `graphqec.qecc.__init__` と code factory に新 code を登録する
3. `get_exp_data()` 用の前処理を追加する
   - `reconstruct_graph_detector_coords.py` の mapping 規則を使って、各 sample を detector-order の二値ベクトルへ戻す
   - `encoding/cycle/readout` の切り分けは `dataloader.py` と同じ規約に合わせる
4. logical label を別ファイルで与える
   - 既存ローダは `logical_flips` を前提にしているため、`obs_flips` 相当が要る

この案では `fullgraph` の edge 自体は学習に使わない。
`fullgraph` は detector-event vector を復元するための中間表現にとどまる。

## `fullgraph` を本当に使う場合の改造

`fullgraph` 辺をそのままモデルに入れたいなら、最小改造では済まない。
必要なのは次。

1. `QuantumCode` / raw-syndrome 前提をやめる
2. `decoder/nn/dataloader.py` を defect-graph loader に差し替える
3. `QECCDecoder._decode()` の raw detector-vector 分割処理を捨てる
   - 現在は `raw_syndromes` を `encoding/cycle/readout` に機械的に split している
   - 参照: `NN-based/graphqec-paper/graphqec/decoder/nn/models.py:42-58`
4. `GraphRNNDecoderV5` / `GraphLinearAttnDecoderV2` の
   `tanner_graph[...].data_to_check`, `data_to_logical` 依存を外し、
   defect-defect edge list ベースの encoder に置き換える

これは事実上、`graphqec-paper` の surface-code backend を再利用しつつ、
NN decoder を別物に差し替える規模の変更である。

## 推奨

`graphqec-paper` に接続するなら、現実的なのは
「`fullgraph` から raw detector-event vector を再構成して既存モデルへ流す」案だけである。
`fullgraph` 辺を主役にしたいなら、この repo は土台としては適していない。

---

## 2. `QEC_GNN-RNN`

## 現状の入力契約

この repo は `Dataset.generate_batch()` が全ての入力を作る。
流れは次。

1. `stim.Circuit.generated(...)` を作る
2. `compile_detector_sampler()` から `detection_events` と `observable_flips` を取る
3. fired detector を node feature に変換する
4. `knn_graph()` で edge を張る
5. `GRUDecoder` に渡す

- 参照: `NN-based/QEC_GNN-RNN/data.py:27-82`
- 参照: `NN-based/QEC_GNN-RNN/data.py:84-110`
- 参照: `NN-based/QEC_GNN-RNN/data.py:169-218`
- 参照: `NN-based/QEC_GNN-RNN/data.py:220-268`
- 参照: `NN-based/QEC_GNN-RNN/gru_decoder.py:33-42`
- 参照: `NN-based/QEC_GNN-RNN/gru_decoder.py:55-82`

モデル側は `x, edge_index, edge_attr, batch_labels, label_map` しか見ていないので、
入力 graph を差し替えやすい。

- 参照: `NN-based/QEC_GNN-RNN/gru_decoder.py:33-42`

### この repo での意味

3 つの候補の中では、`fullgraph` を最も自然に受け入れられる。
現在の `knn_graph()` を `fullgraph` に置き換えればよく、model 本体はほぼ触らずに済む。

## 必要な改造

### 1. 入力モードを追加する

`Args` に、例えば次を追加する。

- `input_mode: Literal["stim", "fullgraph"]`
- `graph_root`
- `mapping_root`
- `label_path`
- `graph_kind` (`full` 固定でもよい)

現状 `Args` は Stim 生成用パラメータしか持っていない。

- 参照: `NN-based/QEC_GNN-RNN/args.py:4-28`

### 2. `Dataset` を二系統に分ける

現状の `Dataset` は `__init__` で必ず circuit を作る。

- 参照: `NN-based/QEC_GNN-RNN/data.py:27-49`

ここを次のどちらかに変える必要がある。

- `Dataset` に `stim` / `fullgraph` の分岐を入れる
- もしくは `StimDataset` と `FullGraphDataset` に分割する

後者の方が安全である。

### 3. `fullgraph` 用の sample loader を追加する

新しい loader は各 JSON について次を作る。

1. `node_features`
   - `reconstruct_graph_detector_coords.py` の CSV から `(x, y, t)` を取得
   - stabilizer type は現行と同じく座標 parity から再計算する
     - 現行コードは `syndrome_mask` で `Z` / `X` を判定している
     - 参照: `NN-based/QEC_GNN-RNN/data.py:76-82`
     - 参照: `NN-based/QEC_GNN-RNN/data.py:196-200`
2. `edge_index`
   - `fullgraph` の `[src, dst, weight]` を PyTorch Geometric 用の edge list に変換
3. `edge_attr`
   - 現行は距離逆二乗を使うが、`fullgraph` では保存済み weight をそのまま使うか、
     単調変換して使う
4. `labels`, `label_map`
   - 最初は「1 sample = 1 graph」にすればよい
   - `sliding` を無効にして、各 sample に 1 個の graph label を割り当てるのが最小
5. `flips`
   - logical label を sidecar から読む

### 4. batch 生成を online sampling から DataLoader 化する

現行 `train_model()` は各 iteration ごとに `dataset.generate_batch()` を呼ぶ。

- 参照: `NN-based/QEC_GNN-RNN/gru_decoder.py:74-82`

`fullgraph` モードでは dataset は有限集合なので、
この部分を `torch.utils.data.DataLoader` ベースへ変える必要がある。
少なくとも次のどちらかが要る。

- `generate_batch()` を「事前に読み込んだ sample 群からランダム mini-batch を返す」実装に置き換える
- もしくは `train_model()` / `test_model()` を DataLoader loop に書き換える

### 5. logical label の供給方法を決める

この repo の loss は `observable_flips` を直接学習する。

- 参照: `NN-based/QEC_GNN-RNN/data.py:84-110`
- 参照: `NN-based/QEC_GNN-RNN/gru_decoder.py:81-84`

したがって、`graph/` 側に label が無いままでは supervised training はできない。
必要なのは例えば次のどちらか。

- 元の生成元から logical flip を別 CSV / JSON に出す
- 目的関数を MWPM imitation に変える

後者にするなら `decoder` の出力 head と loss を変える必要がある。

## 推奨

3 repo の中では、`QEC_GNN-RNN` が最も素直な接続先である。
必要な改造はあるが、model はそのままで、

- node feature
- edge list
- edge weight
- graph-level label

の供給部分を差し替えればよい。

---

## 3. `neural_network_decoder`

## 現状の入力契約

この repo は graph を一切読まない。
SQLite DB に入ったシーケンスを読む。

DB から引いている列は次。

- 通常学習: `events`, `err_signal`, `parity`, `length`
- oversample 時: `events`, `err_signal`, `parities`

- 参照: `NN-based/neural_network_decoder/decoder.py:356-400`
- 参照: `NN-based/neural_network_decoder/decoder.py:493-569`

placeholder も完全に sequence/tensor 前提である。

- `x1`: `[batch, n_steps_net1, dim_syndr]`
- `x2`: `[batch, n_steps_net2, dim_syndr]`
- `fx`: `[batch, dim_fsyndr]`
- `y`: `[batch, 1]`

- 参照: `NN-based/neural_network_decoder/decoder.py:166-207`

`gen_batch()` は `events` を `[steps, dim_syndr]` に reshape し、
`err_signal` を final syndrome increment として読む。

- 参照: `NN-based/neural_network_decoder/decoder.py:408-433`

README でも外部 DB を指定して学習すると書かれている。

- 参照: `NN-based/neural_network_decoder/README.md:7-18`

### この repo での意味

この repo に接続するとは、
`fullgraph` を「sequence DB に戻す」ことを意味する。
`fullgraph` の edge 情報は、そのままではモデルに入らない。

## 必要な改造

### 1. `fullgraph` から `events` を復元する前処理を書く

`reconstruct_graph_detector_coords.py` の mapping CSV を使って、
各 sample を detector-time 配列に戻す必要がある。

必要な処理は次。

1. graph node id から detector index と `t` を得る
2. fired detector を各 time step ごとの二値ベクトルに落とす
3. それを `events` blob として flatten する

ここでの注意点は、mapping 規則自体が推論であることと、
virtual boundary node は detector event ではないこと。

- 参照: `scripts/reconstruct_graph_detector_coords.py:10-15`
- 参照: `scripts/reconstruct_graph_detector_coords.py:229-260`

### 2. `err_signal` をどう作るか決める

この repo は `final syndrome increment` を別入力 `fx` に入れている。

- 参照: `NN-based/neural_network_decoder/decoder.py:177-179`
- 参照: `NN-based/neural_network_decoder/decoder.py:240-260`
- 参照: `NN-based/neural_network_decoder/decoder.py:408-433`

`graph/` JSON には `err_signal` そのものは無いので、次のどちらかが必要。

- circuit 幾何から final readout に相当する detector subset を再構成して `err_signal` を作る
- もしくはモデルを少し改造し、`fx` をゼロベクトル固定で学習する

後者の方が実装は軽いが、元論文系の入力契約からは外れる。

### 3. `parity` / `parities` を別途用意する

この repo の教師信号は parity 1 ビットである。

- 参照: `NN-based/neural_network_decoder/decoder.py:180-182`
- 参照: `NN-based/neural_network_decoder/decoder.py:255-260`

`graph/` には parity が無いので、
sidecar label か元データ生成系からの追加出力が必要である。

### 4. SQLite writer を追加する

現行コードは DB 読み込みしか持たない。

- 参照: `NN-based/neural_network_decoder/decoder.py:356-400`

したがって、接続のためには別スクリプトで

- `data(seed, events, err_signal, parity, length)`  
  もしくは
- `data(seed, events, err_signal, parities)`

の schema を持つ SQLite DB を生成する必要がある。

### 5. `fullgraph` edge を使いたいならモデル改造が別途必要

現状の network は LSTM + feedforward であり、
edge list や edge weight を受ける箇所が無い。

- 参照: `NN-based/neural_network_decoder/decoder.py:240-291`
- 参照: `NN-based/neural_network_decoder/repo_investigation.md:103-118`

したがって `fullgraph` の重みや matching 情報を活かしたいなら、

- `dim_syndr` の feature を増やして hand-crafted summary を入れる
- もしくは network 自体を graph-aware に置き換える

のどちらかが必要になる。

## 推奨

`neural_network_decoder` は「外部データを受ける」点では柔軟だが、
graph を使う repo ではない。
`fullgraph` を接続する場合の実態は、
「defect graph を sequence DB に戻して使う」ことになる。

そのため、この 3 本の中では優先度は `QEC_GNN-RNN` や `graphqec-paper` より低い。

---

## まとめ

### 改造の重さ

1. `QEC_GNN-RNN`
   - 最も現実的
   - `fullgraph` を edge list としてそのまま使いやすい
2. `graphqec-paper`
   - raw detector-event 再構成だけなら可能
   - `fullgraph` 辺を本当に使うなら大改造
3. `neural_network_decoder`
   - SQLite 変換器を作れば接続はできる
   - ただし `fullgraph` edge は基本的に使われない

### 共通して先に決めるべきこと

1. 学習ターゲットを何にするか
   - logical flip
   - MWPM weight
   - MWPM matching imitation
2. `reconstruct_graph_detector_coords.py` の node id -> detector id 対応を
   学習データの正式規約として採用するか
3. `fullgraph` の weight を
   - そのまま edge_attr に使うか
   - 距離 / コストへ変換して使うか

上の 3 点が固まらないと、どの repo でも実装は途中で詰まる。
