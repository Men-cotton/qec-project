# QEC_GNN-RNN リポジトリ調査メモ

本メモは `NN-based/PLANS.md` の指示に従い、「シンドロームグラフからマッチングを出力する過程」に絞って整理する。根拠は repo 内のコードと README のみとし、各項目で `明示的記述`、`推論`、`未確認` を分けて書く。

証拠表記ルール:
- 原則として `ファイルパス + 関数/クラス名` を書く。
- README のように関数/クラスを持たない根拠は `関数/クラス名なし` と書く。

## 0. 要約

- `明示的記述`: この repo で「シンドロームグラフ上のマッチング」を実際に行う経路は `mwmp.py` のみであり、`Stim` の detector error model を `pymatching.Matching.from_detector_error_model(...)` に渡して `decode_batch(...)` するラッパである。
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`mwmp.py:9-30`)
- `明示的記述`: repo が明示的に生成する surface-code タスクは `surface_code:rotated_memory_z` と `surface_code:rotated_memory_x` のみである。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.__init__` (`data.py:27-49`)
- `明示的記述`: GNN+GRU 経路は detector event から独自の k-NN グラフを作るが、出力は「マッチしたノード対」や「補正エッジ列」ではなく logical flip の確率である。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.get_node_features`, `Dataset.get_edges`, `Dataset.generate_batch` (`data.py:169-259`); `NN-based/QEC_GNN-RNN/gru_decoder.py` / `GRUDecoder.forward` (`gru_decoder.py:33-42`)
- `推論`: この repo の主眼は「matching solver の実装」ではなく、「PyMatching による MWPM baseline と、別系統の GNN+GRU decoder の比較」である。
  - 根拠: `NN-based/QEC_GNN-RNN/README.md` / 関数/クラス名なし (`README.md:1-44`); `NN-based/QEC_GNN-RNN/examples/test_nn.py` / 関数/クラス名なし (`examples/test_nn.py:8-28`)

## 1. 対象となるグラフ構造と実装範囲

### 1.1 この repo はシンドロームグラフ上のマッチングを実装またはラップしているか

- `明示的記述`: はい。ただし独自実装ではなく、`Stim` から得た detector error model を `PyMatching` に渡すラップに留まる。
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`mwmp.py:14-23`)
- `明示的記述`: repo が実際に呼んでいる matching solver API は `pymatching.Matching.from_detector_error_model(...)` と `matcher.decode_batch(...)` だけで、repo 自身の matching 手続きは `mwmp.py` には書かれていない。
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`mwmp.py:14-25`)

### 1.2 サポートしている表面符号グラフ

- `明示的記述`: `Dataset.__init__` は `FlipType.BIT` のとき `surface_code:rotated_memory_z`、`FlipType.PHASE` のとき `surface_code:rotated_memory_x` を選ぶ。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `FlipType`, `Dataset.__init__` (`data.py:12-14`, `data.py:27-49`)
- `推論`: repo が前提にしているのは rotated planar surface-code memory task であり、toric code や lattice surgery のような別 family を扱う分岐はない。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.__init__` にコード種別切替が 2 択しかないこと (`data.py:43-46`)
- `未確認`: `surface_code:rotated_memory_z/x` が repo 内の記述だけで XXZZ 配置なのか XZZX 配置なのかは確定できない。repo 内に `XZZX` や `XXZZ` の明示記述はない。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.__init__`; `NN-based/QEC_GNN-RNN/README.md` / 関数/クラス名なし (`data.py:27-49`, `README.md:1-44`)

### 1.3 グラフ次元、境界、制約

- `明示的記述`: 回路生成では `rounds=t` を指定し、detector coordinates から `[x, y, t]` を取り出しているため、repo が実際に処理する detector-event graph は時空間 3D 座標を持つ。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.__init_circuit`, `Dataset.get_node_features` (`data.py:57-74`, `data.py:169-200`)
- `推論`: code-capacity 用の純 2D graph を別実装として切り出してはいない。少なくとも repo 内の生成経路は repeated rounds を前提にしている。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.__init_circuit`, `Dataset.get_node_features` (`data.py:57-74`, `data.py:186-194`)
- `未確認`: boundary node を明示的に表すノード ID、専用データ構造、専用 API は repo 内にない。MWPM 経路で boundary がどう内部表現されるかは `Stim` / `PyMatching` 側に委譲されており、repo からは確認できない。
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` が `detector_error_model(...)` をそのまま渡すだけであること (`mwmp.py:15-16`)
- `明示的記述`: neural 側のグラフは detector event のみをノードにし、boundary node を追加しない。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.get_node_features`, `Dataset.get_edges` (`data.py:182-218`)

### 1.4 Capability Matrix

| 実装経路 | Graph dimension (2D/3D) | Boundary node support | Weighted edges support | Hyperedge support (for correlated/Y errors) | Dynamic graph generation | Parallel matching support |
| --- | --- | --- | --- | --- | --- | --- |
| `mwmp.py` による MWPM wrapper | `明示的記述`: 3D repeated-round detector graph。`未確認`: 2D 専用経路は repo 内にない | `未確認`: repo レベルの boundary node API はない。内部処理は `Stim`/`PyMatching` に委譲 | `明示的記述`: あり。ただし重み計算は repo ではなく外部ライブラリ側 | `明示的記述`: なし。`decompose_errors=True` で graph-like に分解 | `明示的記述`: あり。`Args` から circuit と DEM を実行時生成 | `未確認`: `decode_batch` による batched decode はあるが、並列実行制御コードはない |
| `data.py` + `gru_decoder.py` の neural graph | `明示的記述`: `[x, y, t]` を持つ 3D event graph | `明示的記述`: なし。boundary node を導入しない | `明示的記述`: あり。`edge_attr = 1 / ||delta||^2` | `明示的記述`: なし | `明示的記述`: あり。各 batch の detector event から k-NN graph を再構成 | `非該当`: matching solver ではない |

## 2. グラフ構築とエッジ重みの計算

### 2.1 Code-capacity / Phenomenological / Circuit-level の切り分け

- `明示的記述`: `stim.Circuit.generated(...)` では `after_clifford_depolarization`, `before_round_data_depolarization`, `before_measure_flip_probability`, `after_reset_flip_probability` の 4 種に同じ `error_rate` を入れている。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.__init_circuit` (`data.py:57-66`)
- `推論`: したがって repo のデータ生成は code-capacity 専用でも phenomenological 専用でもなく、回路操作点にノイズを入れる circuit-level に最も近い。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.__init_circuit` (`data.py:57-66`)
- `明示的記述`: measurement error は `before_measure_flip_probability=error_rate` により含まれる。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.__init_circuit` (`data.py:64`)

### 2.2 MWPM 経路でのグラフ構築

- `明示的記述`: MWPM 側は repo 内で隣接リストや CSR を手組みせず、`dataset.circuits[sampler_idx].detector_error_model(decompose_errors=True)` を直接使う。
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`mwmp.py:14-16`)
- `明示的記述`: detector error model は `Dataset.__init_circuit` で生成した `Stim` 回路に依存する。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.__init_circuit`; `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`data.py:57-67`, `mwmp.py:15`)
- `明示的記述`: `decompose_errors=True` を指定しているので、matching に渡るのは分解済みの graph-like error model である。
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`mwmp.py:15`)
- `推論`: 元の回路ノイズに Y 誤りや相関誤り成分が含まれていても、repo の matching 経路はそれらを hyperedge のまま保持せず、分解後の表現に落として PyMatching に委譲している。
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` の `decompose_errors=True` と、repo 内に hypergraph 処理コードがないこと (`mwmp.py:15-16`)
- `未確認`: エッジ重みが対数確率なのか、整数重みなのか、丸めがあるのかは repo 内では分からない。重み計算ロジックは `Stim`/`PyMatching` の内部にあり、この repo には重みを読む・上書きするコードがない。
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`mwmp.py:15-23`)

### 2.3 Neural graph 経路でのグラフ構築

- `明示的記述`: `sample_syndromes` は detector event 行列 `detection_array` を生成し、`get_node_features` は発火した detector の座標を `self.detector_coordinates[sampler_idx][s]` で取り出す。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.sample_syndromes`, `Dataset.get_node_features` (`data.py:84-110`, `data.py:182-200`)
- `明示的記述`: node feature は `[x, y, t, stabilizer_type_Z, stabilizer_type_X]` の 5 次元である。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.get_node_features` (`data.py:171-199`)
- `明示的記述`: `sliding=True` のときは `get_sliding_window` で detector event を時系列 chunk に複製・再配置し、各ノードに `chunk_label` を割り当てる。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.get_sliding_window`, `Dataset.get_node_features` (`data.py:112-167`, `data.py:185-194`)
- `明示的記述`: 辺は `knn_graph(node_features, self.k, batch=labels)` で張る。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.get_edges` (`data.py:202-209`)
- `明示的記述`: neural 側の edge weight は `delta = node_features[dst] - node_features[src]` のノルムを取り、`edge_attr = 1 / ||delta||^2` として保持する。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.get_edges` (`data.py:211-216`)
- `推論`: この重みは物理ノイズ確率から解析的に導いた matching cost ではなく、幾何距離ベースの message-passing 補助特徴である。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.get_edges` に確率モデル参照がないこと (`data.py:202-218`)
- `明示的記述`: X stabilizer と Z stabilizer は `syndrome_mask` から作る 2 ビット feature で同一グラフ内に同居し、type ごとに別 matching graph へ分離しない。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.__init_circuit`, `Dataset.get_node_features`, `Dataset.get_edges` (`data.py:76-82`, `data.py:196-199`, `data.py:202-218`)

## 3. マッチングアルゴリズムの概要

### 3.1 コアアルゴリズム

- `明示的記述`: matching solver は `pymatching.Matching.from_detector_error_model(...)` で作られる MWPM である。
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`mwmp.py:15-16`)
- `明示的記述`: repo の matching 経路では、`PyMatching` 呼び出し以外の solver 手続きは追加されていない。
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`mwmp.py:14-25`)
- `明示的記述`: 実際のデコード呼び出しは `matcher.decode_batch(detection_array)` である。
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`mwmp.py:21-25`)

### 3.2 外部依存と repo 内責務

- `明示的記述`: detector graph の生成は `Stim`、matching は `PyMatching` に依存している。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.__init_circuit` (`data.py:57-74`); `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`mwmp.py:1`, `mwmp.py:15-16`)
- `推論`: repo 自身の責務は「surface-code memory task の回路生成」「detector event のサンプリング」「PyMatching の評価ラップ」であり、matching graph の内部アルゴリズム説明は外部ライブラリに依存する。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset`; `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm`

### 3.3 グラフ理論上の振る舞いとして確認できる範囲

- `明示的記述`: repo が MWPM に渡す観測量は detector の発火パターンだけであり、物理誤りの列や手動構築した path 候補集合は渡していない。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.sample_syndromes` (`data.py:84-110`); `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`mwmp.py:21-23`)
- `明示的記述`: repo 内で消費される MWPM の出力は logical observable prediction だけであり、「マッチされたノード対の一覧」や「選択されたエッジ集合」は取得しない。
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` が `predictions == flips_array` だけを見ること (`mwmp.py:23-25`)
- `推論`: したがってこの repo の公開インターフェース上、「シンドロームグラフからマッチングを出力する過程」は PyMatching のブラックボックス内で完結し、repo 境界で見える出力は最終 observable 判定である。
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`mwmp.py:15-25`)

## 4. 入出力インターフェースとデータ構造

### 4.1 MWPM パイプライン

| 段階 | データ構造 | 明示できる型・形状 | 根拠 |
| --- | --- | --- | --- |
| 回路生成 | `stim.Circuit` | `distance`, `rounds`, 各ノイズ率を持つ外部型 | `NN-based/QEC_GNN-RNN/data.py` / `Dataset.__init_circuit` (`data.py:57-66`) |
| matching graph 入力 | detector error model | 外部型。repo 内では `dataset.circuits[sampler_idx].detector_error_model(decompose_errors=True)` として得る | `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`mwmp.py:15`) |
| syndrome 入力 | `detection_array` | `np.ndarray[bool]`, shape `[batch_size, num_detectors]` | `NN-based/QEC_GNN-RNN/data.py` / `Dataset.sample_syndromes` (`data.py:88-110`) |
| 正解ラベル | `flips_array` | `np.ndarray[np.int32]`, shape `[batch_size, 1]` | `NN-based/QEC_GNN-RNN/data.py` / `Dataset.sample_syndromes` (`data.py:94-110`) |
| solver 出力 | `predictions` | `未確認`: repo では型注釈なし。`flips_array` と比較可能な batch 出力 | `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`mwmp.py:23-25`) |
| 評価値 | `accuracy`, `std` | `torch.Tensor`/数値相当 | `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm`; `NN-based/QEC_GNN-RNN/utils.py` / `standard_deviation` (`mwmp.py:17-30`, `utils.py:101-106`) |

### 4.2 MWPM パイプラインのコア関数シグネチャ

- `明示的記述`: `Dataset.sample_syndromes(self, sampler_idx: int) -> tuple[np.ndarray, np.ndarray]`
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.sample_syndromes` (`data.py:84-110`)
- `明示的記述`: `test_mwpm(dataset: Dataset, n_iter=1000, verbose=True)`
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`mwmp.py:9-30`)
- `未確認`: `pymatching.Matching.from_detector_error_model(...)` と `matcher.decode_batch(...)` の厳密な型は repo では定義していないため、外部 API 詳細は repo 根拠だけでは確定できない。
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`mwmp.py:15-23`)

### 4.3 Neural パイプライン

- `明示的記述`: `Dataset.generate_batch()` は `(node_features, edge_index, labels, label_map, edge_attr, flips)` を返す。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.generate_batch` (`data.py:220-268`)
- `明示的記述`: `node_features` は `[n, 5]`、`edge_index` は adjacency を表す tensor、`edge_attr` は辺重み、`labels` は graph ID、`label_map` は `[batch element, chunk]` 対応、`flips` は logical flip である。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.generate_batch` の docstring (`data.py:224-241`)
- `明示的記述`: `GRUDecoder.forward(self, x, edge_index, edge_attr, batch_labels, label_map)` は最終的に `[batch, 1]` 相当の sigmoid 出力を返す。
  - 根拠: `NN-based/QEC_GNN-RNN/gru_decoder.py` / `GRUDecoder.forward` (`gru_decoder.py:38-42`)
- `明示的記述`: この出力は matching pair list ではなく logical flip probability である。
  - 根拠: `NN-based/QEC_GNN-RNN/gru_decoder.py` / `GRUDecoder.forward`; `NN-based/QEC_GNN-RNN/data.py` / `Dataset.generate_batch` (`gru_decoder.py:38-42`, `data.py:239-241`)

## 5. Neural network 系アルゴリズムの適用

- `明示的記述`: neural network は使われている。ただし matching edge の直接予測にも、PyMatching 用 edge weight の推論にも使っていない。
  - 根拠: `NN-based/QEC_GNN-RNN/gru_decoder.py` / `GRUDecoder`; `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`gru_decoder.py:11-42`, `mwmp.py:14-23`)
- `明示的記述`: GNN 部は `GraphConvLayer` の積み重ねで、各 chunk graph を `global_mean_pool` して graph embedding に変換する。
  - 根拠: `NN-based/QEC_GNN-RNN/gru_decoder.py` / `GRUDecoder.__init__`, `GRUDecoder.embed` (`gru_decoder.py:15-36`); `NN-based/QEC_GNN-RNN/utils.py` / `GraphConvLayer` (`utils.py:41-49`)
- `明示的記述`: `group(...)` は graph embedding を batch ごとの時系列に並べ、欠けた chunk は zero padding して `pack_padded_sequence` に渡す。
  - 根拠: `NN-based/QEC_GNN-RNN/utils.py` / `group` (`utils.py:12-39`)
- `明示的記述`: RNN 部は `nn.GRU` であり、最終隠れ状態を `Linear( hidden_size -> 1 ) + Sigmoid` で logical flip probability に変換する。
  - 根拠: `NN-based/QEC_GNN-RNN/gru_decoder.py` / `GRUDecoder.__init__`, `GRUDecoder.forward` (`gru_decoder.py:22-31`, `gru_decoder.py:38-42`)
- `推論`: これは「matching を解く neural solver」ではなく、「detector-event graph の時系列特徴から logical observable を直接二値分類する decoder」である。
  - 根拠: `NN-based/QEC_GNN-RNN/gru_decoder.py` / `GRUDecoder.forward`; `NN-based/QEC_GNN-RNN/data.py` / `Dataset.generate_batch` (`gru_decoder.py:38-42`, `data.py:239-241`)

## 6. マッチング処理のパフォーマンス・ベンチマーク

- `明示的記述`: `examples/test_nn.py` は学習済み neural decoder と MWPM の両方を `n_iter = 100` で評価する。
  - 根拠: `NN-based/QEC_GNN-RNN/examples/test_nn.py` / 関数/クラス名なし (`examples/test_nn.py:8-28`)
- `明示的記述`: `test_mwpm` が出す評価指標は `accuracy` と binomial 由来の `std`、および `data_time` と `model_time` である。
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm`; `NN-based/QEC_GNN-RNN/utils.py` / `standard_deviation` (`mwmp.py:17-30`, `utils.py:101-106`)
- `明示的記述`: `GRUDecoder.test_model` も同様に `accuracy`、`std`、`data_time`、`model_time` を評価する。
  - 根拠: `NN-based/QEC_GNN-RNN/gru_decoder.py` / `GRUDecoder.test_model` (`gru_decoder.py:136-157`)
- `明示的記述`: 学習時ログ `TrainingLogger` は epoch ごとの `model_time`, `data_time`, `loss`, `accuracy` などを保存する。
  - 根拠: `NN-based/QEC_GNN-RNN/utils.py` / `TrainingLogger.on_epoch_end`, `TrainingLogger.on_training_end` (`utils.py:65-99`)
- `推論`: ただしこれらは「matching solver としての scaling benchmark」ではなく、decoder 全体の論理精度評価と粗い処理時間計測である。graph size に対する scaling、メモリ使用量、matching cost optimality、exact solver との差分は測っていない。
  - 根拠: `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm`; `NN-based/QEC_GNN-RNN/gru_decoder.py` / `GRUDecoder.test_model`; `NN-based/QEC_GNN-RNN/utils.py` / `TrainingLogger` に該当指標がないこと (`mwmp.py:9-30`, `gru_decoder.py:136-157`, `utils.py:51-99`)

## 7. 最終結論

- `明示的記述`: この repo における「シンドロームグラフからマッチング出力」経路は、`Stim` が生成した repeated-round rotated surface-code 回路から detector error model を作り、`PyMatching` の MWPM に `decode_batch` させる wrapper である。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.__init_circuit`; `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`data.py:57-74`, `mwmp.py:14-23`)
- `明示的記述`: repo 内で観測できる matching の入出力は「入力: detector event の bool 配列」「出力: logical observable prediction」であり、matching edge や correction path は公開していない。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.sample_syndromes`; `NN-based/QEC_GNN-RNN/mwmp.py` / `test_mwpm` (`data.py:84-110`, `mwmp.py:21-25`)
- `明示的記述`: neural GNN+GRU 経路は別物であり、matching graph solver ではなく detector-event graph から logical flip を直接分類する decoder である。
  - 根拠: `NN-based/QEC_GNN-RNN/data.py` / `Dataset.get_edges`, `Dataset.generate_batch`; `NN-based/QEC_GNN-RNN/gru_decoder.py` / `GRUDecoder.forward` (`data.py:202-259`, `gru_decoder.py:33-42`)
