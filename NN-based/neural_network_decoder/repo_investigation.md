# Repo Investigation for `neural_network_decoder`

## 調査スコープ

- 明示的記述:
  - 現在の `NN-based/neural_network_decoder` には [README.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/README.md:1), [data_format.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/data_format.md:1), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:1), [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:1), [src/qec_functions.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/qec_functions.py:1), notebooks がある。根拠: `find NN-based/neural_network_decoder -maxdepth 3 -type f`。関数/クラス名: `Decoder`, `Data`, `calc_plog`, `calc_stats`。
  - 旧トップレベルの `decoder.py` は現在の tree には存在せず、decoder の主実装は [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:1) にある。根拠: `find NN-based/neural_network_decoder -maxdepth 3 -type f`。関数/クラス名: `Decoder`。
- 推論:
  - 本メモは current tree のファイルだけを根拠に記述する。
  - 以前の単一ファイル版ではなく、現在は `src/` と `data_format.md` を含む別レイアウトの repo 状態である。

## 1. 対象となるグラフ構造と実装範囲

### 1-1. シンドロームグラフ上のマッチング処理を実装またはラップしているか

- 結論:
  - この repo は、シンドロームグラフ上のマッチング処理を実装もラップもしていない。
  - 現在の実装は、外部で生成された syndrome increment 列と final syndrome increment を neural network に入力し、logical bit-flip parity を直接予測する decoder である。

- 明示的記述:
  - `README.md` は「A decoder for small topological surface and color codes... based on a combination of recurrent and feedforward neural networks」と記述する。根拠: [README.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/README.md:3)。関数/クラス名: 該当なし。
  - `src/decoder.py` の `Decoder` docstring は、LSTM layers と 2 つの feedforward head を持つ neural network と説明する。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:22)。関数/クラス名: `Decoder`。
  - `src/decoder.py` の import は `numpy`, `tensorflow`, `tensorflow.contrib.rnn` と `qec_functions.calc_plog` であり、matching solver や graph library を import しない。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:13), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:19)。関数/クラス名: `Decoder`。
  - `src/database_io.py` は DB から `syndrome_increments`, `final_syndr_incr`, `parity_of_bitflips`, `no_cycles` を読む。ノード集合やエッジ集合は扱わない。根拠: [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:77), [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:131), [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:159)。関数/クラス名: `Data.gen_batches`。
  - `Decoder.test_net` は parity prediction と真値 parity を比較して `comparison` を返すが、matching pair や recovery edge list は返さない。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:502), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:524)。関数/クラス名: `Decoder.test_net`。

- 推論:
  - 本 repo の入出力は「syndrome graph -> matching result」ではなく「syndrome time-series tensor -> parity classification」である。
  - したがって PLANS.md の matching 固有項目は、多くが「未実装」よりも「この repo の責務外」と整理するのが妥当である。

### 1-2. どのような surface/code family を前提またはサポートしているか

- 明示的記述:
  - `README.md` は「small topological surface and color codes, and potentially other stabilizer codes, when encoding a single logical qubit」と述べる。根拠: [README.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/README.md:3)。関数/クラス名: 該当なし。
  - `Decoder` docstring も `surface codes or the color code` を例示する。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:25)。関数/クラス名: `Decoder`。
  - `Decoder.__init__` は `code_distance`, `dim_syndr`, `dim_fsyndr` を明示的に受け取る。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:58)。関数/クラス名: `Decoder.__init__`。

- 推論:
  - 対応 family は surface code と color code まで広がっている。
  - ただし rotated planar / toric / XXZZ / XZZX のような surface-code graph variant は repo 内に明示されておらず未確認である。
  - 単一 logical qubit 前提は明示されているため、multi-logical-qubit patch や lattice surgery を直接扱う設計ではない。

### 1-3. Capability Matrix

| Graph dimension (2D/3D) | Boundary node support | Weighted edges support | Hyperedge support (for correlated/Y errors) | Dynamic graph generation | Parallel matching support |
| --- | --- | --- | --- | --- | --- |
| 未確認。repeated syndrome cycles は扱うが、graph 表現が無いため 2D/3D matching graph としては判定不能。根拠: [README.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/README.md:10), [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:91)。関数/クラス名: `Data.gen_batches` | 非対応。boundary node を表すデータ構造や API がない。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:127), [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:131)。関数/クラス名: `Decoder._init_network_variables`, `Data.gen_batches` | 非対応。edge weight や matching cost を生成・保持しない。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:163), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:502)。関数/クラス名: `Decoder._init_network_functions`, `Decoder.test_net` | 非対応。hyperedge を表す構造がない。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:261), [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:207)。関数/クラス名: `Decoder.network`, `Data._gen_batch_oversample` | 非対応。graph generator はなく、外部生成済み SQLite dataset を読む。根拠: [README.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/README.md:10), [data_format.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/data_format.md:4), [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:38)。関数/クラス名: `Data.__init__`, `Data.load_database` | 非対応。matching solver 自体がない。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:502)。関数/クラス名: `Decoder.test_net` |

## 2. グラフ構築とエッジ重みの計算

### 2-1. シンドロームグラフがどのように構築されるか

- 明示的記述:
  - `data_format.md` は dataset schema を定義し、training/validation では `syndrome_increments: N_max x N_syndr`, `final_syndr_incr: N_fsyndr`, `parity_of_bitflips: size 1`, `no_cycles` を持つと記述する。根拠: [data_format.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/data_format.md:6)。関数/クラス名: 該当なし。
  - test dataset では `syndrome_increments: N_max x N_syndr`, `final_syndr_incr: N_max x N_fsyndr`, `parity_of_bitflips: N_max` を持つ。根拠: [data_format.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/data_format.md:17)。関数/クラス名: 該当なし。
  - `Data._gen_batch` は `syndrome_increments` を bool 配列へ復元し、`syndr_incr = syndr_incr[:len_max]` で sequence tensor を作る。根拠: [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:207), [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:218)。関数/クラス名: `Data._gen_batch`。
  - `Data._gen_batch_oversample` は test row を複数 cycle 長のサンプルへ展開する。根拠: [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:225), [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:256)。関数/クラス名: `Data._gen_batch_oversample`。

- 推論:
  - repo 内で構築されるのは graph ではなく、前処理済み syndrome increment tensor である。
  - graph node 生成、edge 接続、boundary connection は dataset 生成側の暗黙仕様に埋め込まれており、この repo では再構築されない。

### 2-2. Code capacity / Phenomenological / Circuit-level の違いの反映

- 明示的記述:
  - `README.md` は Ref. [3] について「color codes with circuit level noise」と述べる。根拠: [README.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/README.md:21)。関数/クラス名: 該当なし。
  - `README.md` は input data の前処理が「depends on the quantum circuit and is described in [3]」と述べる。根拠: [README.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/README.md:10)。関数/クラス名: 該当なし。
  - `data_format.md` は repeated cycles を前提に `no_cycles` と cycle-by-cycle test labels を定義する。根拠: [data_format.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/data_format.md:4), [data_format.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/data_format.md:17)。関数/クラス名: 該当なし。

- 推論:
  - repeated measurement rounds を含む phenomenological / circuit-level 系のデータは扱える。
  - しかし repo 内コードは noise model ごとに graph topology を切り替えない。差分はすべて外部生成済み dataset とその preprocessing に押し込まれている。

### 2-3. エッジ重みと相関誤り

- 明示的記述:
  - `src/decoder.py` は LSTM 出力と final syndrome increment を feedforward head に入れるが、edge weight を計算しない。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:303), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:317)。関数/クラス名: `Decoder.network`。
  - `src/qec_functions.py` は fidelity decay fit と logical error rate 推定を行うが、matching weight は扱わない。根拠: [src/qec_functions.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/qec_functions.py:59), [src/qec_functions.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/qec_functions.py:95)。関数/クラス名: `calc_plog`, `calc_stats`。

- 推論:
  - Y 誤りや相関誤りを graph の hyperedge として表現する設計ではない。
  - 相関情報が使われる場合も、sequence の統計パターンとして neural network に吸収させる方式である。

## 3. マッチングアルゴリズムの概要

### 3-1. コアアルゴリズム

- 結論:
  - MWPM, Union-Find, 貪欲 matching, tensor-network matching は実装されていない。
  - コアアルゴリズムは、variable-length syndrome sequence を読む LSTM と、main/auxiliary 2 head の feedforward classifier による parity prediction である。

- 明示的記述:
  - `Decoder.network` は `tf.nn.dynamic_rnn` で sequence を処理し、最後の LSTM 出力を抽出する。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:261), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:279)。関数/クラス名: `Decoder.network`。
  - main head は `last_lstm_out` と `input_fsyndr` を結合し、auxiliary head は `last_lstm_out` のみを使う。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:292), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:313)。関数/クラス名: `Decoder.network`。
  - `Decoder._init_network_functions` は `self.predictions` と `self.predictions_aux` を `sigmoid` で作り、cross-entropy と L2 regularization を含む cost を最適化する。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:159), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:169), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:182), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:238)。関数/クラス名: `Decoder._init_network_functions`。

- 推論:
  - アルゴリズム分類としては graph-theoretic decoder ではなく sequence-model-based neural decoder である。

### 3-2. 外部ライブラリ依存

- 明示的記述:
  - `README.md` は TensorFlow ベースであると明記する。根拠: [README.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/README.md:3)。関数/クラス名: 該当なし。
  - `src/qec_functions.py` は `scipy.optimize` と `matplotlib` を使う。根拠: [src/qec_functions.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/qec_functions.py:13), [src/qec_functions.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/qec_functions.py:14)。関数/クラス名: `calc_plog`, `calc_stats`, `plot_fids`。

- 推論:
  - 依存は neural network 学習と統計評価用であり、PyMatching や Blossom のような matching solver 依存はない。

## 4. 入出力インターフェースとデータ構造

### 4-1. 入力データ（グラフ）

- 明示的記述:
  - dataset schema は SQLite table `data` と optional な `info` table である。根拠: [data_format.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/data_format.md:4), [data_format.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/data_format.md:26)。関数/クラス名: 該当なし。
  - `Data.load_database` は sqlite3 connection を開いて `db_dict` に保存する。根拠: [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:47)。関数/クラス名: `Data.load_database`。

- 推論:
  - 入力「グラフ」は存在しない。データ構造は graph object ではなく SQLite row から復元される numpy bool array である。

### 4-2. 入力データ（シンドローム）

- 明示的記述:
  - `Decoder._init_network_variables` は `x: [None, None, dim_syndr]`, `fx: [None, dim_fsyndr]`, `y: [None, 1]`, `length: [None]` を定義する。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:123), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:130), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:133), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:136)。関数/クラス名: `Decoder._init_network_variables`。
  - `Data.gen_batches` の batch 出力は `seeds, syndrome increments, final syndrome increments, number of cycles, final parities` である。根拠: [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:102)。関数/クラス名: `Data.gen_batches`。
  - `Data._gen_batch` と `Data._gen_batch_oversample` は SQLite の BLOB/byte 列を `np.fromstring(..., dtype=bool)` で bool array に戻す。根拠: [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:218), [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:248), [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:249), [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:250)。関数/クラス名: `Data._gen_batch`, `Data._gen_batch_oversample`。

- 推論:
  - solver に defect node ID list を渡すのではなく、各 cycle の syndrome increment bit-vector と final syndrome increment vector を渡す。

### 4-3. 出力データ

- 明示的記述:
  - `self.predictions` と `self.predictions_aux` は parity が odd である確率である。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:169), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:170)。関数/クラス名: `Decoder._init_network_functions`。
  - `Decoder.test_net` は `preds = np.around(preds).astype(bool)` で parity を二値化し、正誤比較結果を step ごとの `comparison` にまとめる。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:515), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:522), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:524)。関数/クラス名: `Decoder.test_net`。
  - `calc_plog` と `calc_stats` は `comparison` から `plog`, `steps`, `fids`, `x0` などを返す。根拠: [src/qec_functions.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/qec_functions.py:59), [src/qec_functions.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/qec_functions.py:95), [src/qec_functions.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/qec_functions.py:174)。関数/クラス名: `calc_plog`, `calc_stats`。

- 推論:
  - 出力は matching pair list, correction edge set, Pauli frame update ではなく、logical parity prediction とその統計評価である。

### 4-4. コア関数・クラスのシグネチャ

- 明示的記述:
  - `Decoder.__init__(self, code_distance, dim_syndr, dim_fsyndr, lstm_iss, ff_layer_sizes, checkpoint_path, keep_prob=1, aux_loss_factor=1, l2_prefactor=0)`。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:58)。関数/クラス名: `Decoder.__init__`。
  - `Data.__init__(self, training_fname, validation_fname, test_fname, verbose=False, store_in_memory=False)`。根拠: [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:31)。関数/クラス名: `Data.__init__`。
  - `Data.gen_batches(self, n_batches, batch_size, db_type, len_buffered, len_min=None, len_max=None, step_list=None, select_random=True)`。根拠: [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:77)。関数/クラス名: `Data.gen_batches`。
  - `Decoder.train_one_epoch(self, train_batches, learning_rate=0.001)`。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:404)。関数/クラス名: `Decoder.train_one_epoch`。
  - `Decoder.calc_feedback(self, batches, validation=True)`。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:426)。関数/クラス名: `Decoder.calc_feedback`。
  - `Decoder.test_net(self, batches, auxillary=False)`。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:462)。関数/クラス名: `Decoder.test_net`。

- 推論:
  - 型注釈は無いが、戻り値は次のように整理できる。

```python
Data.gen_batches(...)
-> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]

Decoder.test_net(batches, auxillary=False)
-> list[list[bool]]

Decoder.calc_feedback(batches, validation=True)
-> float

calc_plog(data)
-> float

calc_stats(data, bootstrap, p0, ...)
-> dict[str, object]
```

## 5. Neural network 系アルゴリズムの適用

### 5-1. NN が使われている箇所

- 結論:
  - 使われている。
  - ただし使い方は「edge weight を推論して matching に渡す」ではなく、「syndrome sequence から logical parity を直接予測する」である。

- 明示的記述:
  - `Decoder.network` は variable-length syndrome sequence を LSTM で処理する。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:261), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:276)。関数/クラス名: `Decoder.network`。
  - main head は final syndrome increment を併用し、auxiliary head は時間方向の translation invariance を促す目的と説明される。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:293), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:314)。関数/クラス名: `Decoder.network`。

- 推論:
  - GNN や matching-edge classifier ではなく、sequence model + MLP head の構成である。

### 5-2. テンソル化の方法

- 明示的記述:
  - `x` は `[batch, steps, dim_syndr]`、`fx` は `[batch, dim_fsyndr]` として TensorFlow placeholder に入る。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:127), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:130)。関数/クラス名: `Decoder._init_network_variables`。
  - `Data.gen_batches` は SQLite row を numpy array に変換して `arrX`, `arrFX`, `arrL`, `arrY` を yield する。根拠: [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:173), [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:199)。関数/クラス名: `Data.gen_batches`。

- 推論:
  - テンソル化単位は graph node / edge ではなく、cycle ごとの syndrome increment bit vector である。

## 6. マッチング処理のパフォーマンス・ベンチマーク

### 6-1. matching solver としての性能評価があるか

- 結論:
  - ない。

- 明示的記述:
  - `README.md` は training notebook と evaluation notebook による decoder 評価を説明する。根拠: [README.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/README.md:10)。関数/クラス名: 該当なし。
  - `Decoder.calc_feedback` は `calc_plog(self.test_net(...))` により logical error rate を計算する。根拠: [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:442), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:444)。関数/クラス名: `Decoder.calc_feedback`。
  - `calc_stats` は fidelity decay fit と bootstrapping による `plog` / `x0` / error bar を返す。根拠: [src/qec_functions.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/qec_functions.py:95), [src/qec_functions.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/qec_functions.py:174)。関数/クラス名: `calc_stats`。

- 推論:
  - 評価対象は matching runtime や matching cost 最適性ではなく、logical fidelity decay と logical error rate である。
  - matching solver が存在しないため、matching-specific benchmark も存在しない。

## 総括

- 明示的記述:
  - current repo は、surface code と color code を含む単一 logical qubit 向け neural decoder を提供し、実装本体は [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:1)、データ入出力は [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:1)、dataset schema は [data_format.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/data_format.md:1) にある。根拠: [README.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/README.md:3), [src/decoder.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/decoder.py:22), [src/database_io.py](/home/mencotton/qec-project/NN-based/neural_network_decoder/src/database_io.py:19), [data_format.md](/home/mencotton/qec-project/NN-based/neural_network_decoder/data_format.md:1)。関数/クラス名: `Decoder`, `Data`。
- 推論:
  - この repo は、シンドロームグラフから matching を計算する decoder ではない。
  - 代わりに、外部で生成・前処理済みの syndrome increment sequence を入力とする sequence-model-based decoder であり、出力は logical parity prediction と logical error rate 統計である。
  - そのため、boundary node support, weighted edges, hyperedge matching, MWPM backend といった PLANS.md の matching-centric capability は、現状「未確認」ではなく「対象外」と判断できる項目が多い。
