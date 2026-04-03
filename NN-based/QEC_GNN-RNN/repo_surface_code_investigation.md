# QEC_GNN-RNN リポジトリ調査メモ

本メモは、リポジトリ内のコードとドキュメントのみを根拠として整理する。各節で `明示的記述` と `推論` を分離し、証拠が見つからない場合は `未確認` と明記する。

証拠表記のルール:
- 可能な限り `ファイルパス + 関数/クラス名` を書く。
- README やトップレベル script のように関数/クラスを持たない根拠は、`関数/クラス名なし` と明記して扱う。

## 1. Surface code の対応状況と実装範囲

### 結論

- `明示的記述`: この repo は surface code の QEC を対象としている。README は「`decoding the surface code`」と記載し、`Dataset` は `Stim` の生成タスクとして `surface_code:rotated_memory_z` または `surface_code:rotated_memory_x` を選ぶ。
  - 根拠: `README.md` / 見出し・本文, `Dataset.__init__`
  - 参照: `README.md:1-3`, `data.py:27-49`
- `明示的記述`: 実装の中心ディレクトリ/ファイルはルート直下の `data.py`, `gru_decoder.py`, `mwmp.py`, `utils.py`, `args.py` と、運用例を置く `examples/`, 学習済み重みを置く `models/` である。
  - 根拠: `README.md` の各ファイル説明（関数/クラス名なし）
  - 参照: `README.md:9-35`

### 対応するコード種別

- `明示的記述`: コード生成タスクとして repo 内で明示的に使われているのは `surface_code:rotated_memory_z` と `surface_code:rotated_memory_x` の 2 種だけである。
  - 根拠: `FlipType.BIT` / `FlipType.PHASE` に応じて `Dataset.__init__` が `self.code_task` を 2 択で設定している。
  - 参照: `data.py:12-14`, `data.py:27-49`
- `推論`: これは「rotated planar surface-code memory task」を対象とする実装であり、少なくとも repo 内に `XZZX`, `XXZZ`, `toric`, `lattice surgery` など別バリアントを切り替える設定・分岐は存在しない。
  - 根拠: `Dataset.__init__` の 2 択以外のコード種別パラメータが存在しないこと、repo 全体検索で `XZZX` / `XXZZ` / `lattice` が実装名として現れないこと。
  - 参照: `data.py:27-49`
- `未確認`: `surface_code:rotated_memory_z/x` が具体的に XXZZ 配置か XZZX 配置かを repo 自身は説明していない。repo 内にその明示記述は見つからない。
  - 根拠: `README.md`（関数/クラス名なし）, `Dataset` docstring, `Dataset.__init__`
  - 参照: `README.md:1-35`, `data.py:17-26`

### 実装上の制約

- `明示的記述`: 単一 logical qubit 前提である。`sample_syndromes` は observables を `[b, 1]` で返し、`GRUDecoder` も最終出力を 1 次元 sigmoid に固定している。
  - 根拠: `Dataset.sample_syndromes`, `GRUDecoder.__init__`, `GRUDecoder.forward`
  - 参照: `data.py:84-110`, `gru_decoder.py:22-31`, `gru_decoder.py:38-42`
- `推論`: multi-logical-qubit は未対応というより、現状アーキテクチャ上の対象外である。理由は、データ生成側が 1 observable のみを教師信号として返し、モデル側も 1 ビット予測しか持たないため。
  - 根拠: `flips_array` の shape 記述、decoder 出力層、`mwpm.py` の比較方法
  - 参照: `data.py:94-110`, `gru_decoder.py:22-25`, `mwmp.py:21-25`
- `明示的記述`: patch は単一の `distance` スカラーで生成され、安定化子数を `distance ** 2 - 1` として扱う。縦横別距離や複数パッチを表す引数は存在しない。
  - 根拠: `Args.distance`, `Dataset.__init__`
  - 参照: `args.py:8-15`, `data.py:27-38`
- `推論`: rectangular patch は「単に例がない」だけでなく、現行のデータ構造・引数設計の外にある。理由は距離が 1 変数で、`syndrome_mask` も `(distance + 1) x (distance + 1)` の正方格子を前提に構築されているため。
  - 根拠: `Dataset.__init__`, `Dataset.__init_circuit`
  - 参照: `data.py:33-38`, `data.py:76-82`
- `推論`: boundary は planar/open boundary とみなすのが自然だが、repo 自体は boundary 種別を明示説明していない。少なくとも periodic boundary を選ぶ実装・パラメータは存在しない。
  - 根拠: `surface_code:rotated_memory_z/x` という有限パッチ用のタスク名、有限サイズ `syndrome_mask`
  - 参照: `data.py:43-46`, `data.py:76-82`
- `明示的記述`: repeated syndrome rounds は対応している。`Stim` 回路生成時に `rounds=t` を渡し、時刻座標 `t` を node feature に含め、スライディング窓で時系列 chunk に分割し、GRU に入力している。
  - 根拠: `Dataset.__init_circuit`, `Dataset.get_node_features`, `Dataset.get_sliding_window`, `GRUDecoder.forward`, `utils.group`
  - 参照: `data.py:57-67`, `data.py:112-200`, `gru_decoder.py:38-42`, `utils.py:12-39`
- `明示的記述`: odd distance 制約は repo 内コードでは検証していない。`Args.distance` や `Dataset` に odd-only assertion はない。
  - 根拠: `Args`, `Dataset.__init__`
  - 参照: `args.py:8-15`, `data.py:27-49`
- `推論`: 学習済み重みと examples は `distance=3,5,7` のみを使っており、利用実績は odd distance に偏っている。ただし even distance が実行可能かどうかは repo 内証拠だけでは確定できない。
  - 根拠: `examples/load_nn.py`（関数/クラス名なし）, `examples/train_nn.py`（関数/クラス名なし）, `models/` のファイル名
  - 参照: `examples/load_nn.py:7-62`, `examples/train_nn.py:10-22`

### Capability Matrix

| code family | patch shape | single/multi logical qubit | boundaries | odd distance restriction | repeated syndrome rounds | measurement error support | active correction support | lattice surgery | benchmark scripts | neural decoder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `surface_code:rotated_memory_z` | `推論`: rotated square planar patch | `明示的記述`: single logical qubit | `推論`: open / planar boundary | `未確認`: repo 内に odd-only 制約なし | `対応` | `対応` | `非対応` | `非対応` | `対応` | `対応` |
| `surface_code:rotated_memory_x` | `推論`: rotated square planar patch | `明示的記述`: single logical qubit | `推論`: open / planar boundary | `未確認`: repo 内に odd-only 制約なし | `対応` | `対応` | `非対応` | `非対応` | `対応` | `対応` |

### Matrix 各項目の根拠

- `measurement error support = 対応`
  - `明示的記述`: `Stim` 回路生成で `before_measure_flip_probability=error_rate` を有効化している。
  - 根拠: `Dataset.__init_circuit`
  - 参照: `data.py:57-66`
- `active correction support = 非対応`
  - `明示的記述`: NN も MWPM も出力は logical flip/observable の予測精度評価であり、物理 qubit への補正列や回路フィードバック API はない。
  - 根拠: `Dataset.sample_syndromes`, `GRUDecoder.forward`, `GRUDecoder.test_model`, `test_mwpm`
  - 参照: `data.py:84-110`, `gru_decoder.py:38-42`, `gru_decoder.py:136-157`, `mwmp.py:9-30`
  - `推論`: これは単なる未実装というより、現在の評価対象が memory/readout 実験の logical observable 判定に固定されているため、設計対象外に近い。
- `lattice surgery = 非対応`
  - `明示的記述`: 該当 API, 回路生成分岐, 複数パッチ結合処理が repo 内に存在しない。
  - 根拠: `Dataset.__init__` が 2 つの memory task のみを扱うこと、repo 全体検索
  - 参照: `data.py:27-49`
- `benchmark scripts = 対応`
  - `明示的記述`: `examples/test_nn.py` が NN と MWPM の両方を評価し、`mwmp.py` が MWPM ベンチマーク関数を提供する。
  - 根拠: `examples/test_nn.py`, `mwmp.py`
  - 参照: `examples/test_nn.py:8-28`, `mwmp.py:9-30`
- `neural decoder = 対応`
  - `明示的記述`: `GRUDecoder` が GNN + GRU を実装し、学習・評価 API を持つ。
  - 根拠: `GRUDecoder`
  - 参照: `gru_decoder.py:11-157`

## 2. 対象ノイズモデル

### 2.1 デコーダ実装が仮定するノイズモデル

#### MWPM

- `明示的記述`: `test_mwpm` は `dataset.circuits[sampler_idx].detector_error_model(decompose_errors=True)` から detector error model を作り、`pymatching.Matching.from_detector_error_model(...)` に直接渡している。
  - 根拠: `test_mwpm`
  - 参照: `mwmp.py:9-16`
- `結論`: MWPM 実装が前提にしているのは、`Dataset` が生成した `Stim` 回路から導かれる detector error model である。
  - `明示的記述`: detector error model は `Dataset.__init_circuit` が作る `surface_code:rotated_memory_z/x` 回路に依存する。
  - 根拠: `Dataset.__init_circuit`, `test_mwpm`
  - 参照: `data.py:51-67`, `mwmp.py:14-16`
- `推論`: 前提モデルの分類は `circuit-level` が最も近い。理由は、回路生成時に data, Clifford, measurement, reset の各位置にノイズ確率を入れており、その回路から detector error model を抽出しているため。
  - 根拠: `Stim` 回路生成引数
  - 参照: `data.py:57-66`
- `明示的記述`: `decompose_errors=True` を使っており、MWPM に渡す表現は detector error model 上の分解済み誤りである。
  - 根拠: `test_mwpm`
  - 参照: `mwmp.py:15`
- `推論`: 相関誤りが元の回路に含まれていても、matching 側では分解後の graph-like error model に落として扱う設計であり、高次相関をそのまま保持する実装にはなっていない。
  - 根拠: `decompose_errors=True`、追加の相関処理コード不在
  - 参照: `mwmp.py:15-16`
- `measurement error の扱い`
  - `明示的記述`: underlying circuit には `before_measure_flip_probability` が含まれるため、MWPM は measurement error を含む detector error model を前提にしている。
  - 根拠: `Dataset.__init_circuit`
  - 参照: `data.py:62-65`
- `データ誤りのみか`
  - `明示的記述`: いいえ。data 以外に Clifford, measurement, reset の誤りも同時に含む。
  - 根拠: `Dataset.__init_circuit`
  - 参照: `data.py:62-65`
- `Y 誤り・相関誤りの扱い`
  - `未確認`: repo 内 docstring は Y 誤りを名指しで説明していない。
  - `推論`: `after_clifford_depolarization` と `before_round_data_depolarization` を用いるため、X/Z のみの純粋独立モデルではなく depolarizing 系の Pauli 誤りを含む前提である可能性が高い。ただし repo 自体は Y の寄与を個別に可視化・分解しない。
  - 根拠: `Dataset.__init_circuit`, `test_mwpm`
  - 参照: `data.py:57-66`, `mwmp.py:15-16`
- `biased noise の前提`
  - `明示的記述`: バイアス比を指定する引数はなく、4 種のノイズ注入確率には同じ `error_rate` を使う。
  - 根拠: `Args.error_rates`, `Dataset.__init_circuit`
  - 参照: `args.py:8-15`, `data.py:55-66`
  - `結論`: biased-noise 専用実装ではない。

#### GNN + GRU decoder

- `明示的記述`: `GRUDecoder` 本体は detector error model や解析的重み計算を持たず、`Dataset.generate_batch()` が返す node features / edges / logical flip label を学習・推論に使う。
  - 根拠: `GRUDecoder.train_model`, `GRUDecoder.test_model`, `GRUDecoder.forward`
  - 参照: `gru_decoder.py:38-42`, `gru_decoder.py:44-135`, `gru_decoder.py:136-157`
- `結論`: NN decoder ロジック自体には `code capacity / phenomenological / circuit-level` の解析的仮定は埋め込まれていない。実際の前提分布は `Dataset` が生成する学習・評価データに依存する。
  - 根拠: `GRUDecoder` が `Dataset` 依存であること
  - 参照: `gru_decoder.py:55-56`, `gru_decoder.py:146-148`
- `推論`: 実運用上は `Dataset` に依存するため、NN decoder も circuit-level ノイズ分布に対して学習・評価される。
  - 根拠: `Dataset.__init_circuit` のノイズ注入方法と `GRUDecoder` のデータ依存
  - 参照: `data.py:57-66`, `gru_decoder.py:55-56`
- `measurement error の扱い`
  - `明示的記述`: 入力は repeated rounds の detector events であり、これらは measurement error を含む `Stim` サンプルから生成される。
  - 根拠: `Dataset.sample_syndromes`, `Dataset.get_node_features`
  - 参照: `data.py:84-110`, `data.py:169-200`
- `データ誤りのみか`
  - `明示的記述`: いいえ。学習データ生成では data / Clifford / measurement / reset の各誤りを同時に注入する。
  - 根拠: `Dataset.__init_circuit`
  - 参照: `data.py:57-66`
- `Y 誤り・相関誤りの扱い`
  - `明示的記述`: NN 入力には誤り種別ラベルそのものはなく、detector event 座標と stabilizer type のみを渡す。
  - 根拠: `Dataset.get_node_features`
  - 参照: `data.py:169-200`
  - `推論`: したがって Y 誤りや相関誤りが存在しても、それらは detection-event pattern に潰された形でのみ学習され、個別カテゴリとしては扱われない。
- `biased noise の前提`
  - `明示的記述`: bias パラメータは存在せず、訓練データは単一の `error_rate` を共通使用する。
  - 根拠: `Args.error_rates`, `Dataset.__init_circuit`
  - 参照: `args.py:8-15`, `data.py:55-66`

### 2.2 ベンチマーク / シミュレーションスクリプトが実際に使うノイズモデル

- `明示的記述`: `Dataset.__init_circuit` は各 `(error_rate, t)` 組について `stim.Circuit.generated(...)` を呼び、以下 4 箇所へ同じ `error_rate` を入れる。
  - `after_clifford_depolarization`
  - `before_round_data_depolarization`
  - `before_measure_flip_probability`
  - `after_reset_flip_probability`
  - 根拠: `Dataset.__init_circuit`
  - 参照: `data.py:55-66`
- `結論`: ベンチマーク/シミュレーションで実際に使うノイズモデルは `circuit-level` である。
  - `理由`: データ誤りだけでなく、各ラウンドの測定、reset、Clifford 操作にもノイズが注入されるため。
- `measurement error の有無`
  - `明示的記述`: あり。`before_measure_flip_probability` を設定している。
  - 根拠: `Dataset.__init_circuit`
  - 参照: `data.py:64`
- `データ誤りのみか`
  - `明示的記述`: いいえ。`before_round_data_depolarization` に加え、`after_clifford_depolarization` と `after_reset_flip_probability` も使う。
  - 根拠: `Dataset.__init_circuit`
  - 参照: `data.py:62-65`
- `相関誤りや Y 誤りの扱い`
  - `未確認`: repo 内テキストは Pauli 成分の内訳や相関の程度を明示していない。
  - `推論`: depolarization を使う以上、純粋な X-only/Z-only ノイズではなく、より一般の Pauli 誤りを含む設定を意図していると読める。ただし repo はその分解統計を出力しない。
- `biased noise の前提`
  - `明示的記述`: なし。error bias を指定する API はなく、examples も単一スカラー `error_rates=[...]` のみ使う。
  - 根拠: `Args`, `examples/train_nn.py`, `examples/test_nn.py`, `examples/load_nn.py`
  - 参照: `args.py:8-15`, `examples/train_nn.py:10-22`, `examples/test_nn.py:10-21`, `examples/load_nn.py:11-60`

### 2.3 code capacity / phenomenological / circuit-level の切り分け

- `code capacity`: `非対応`
  - `明示的記述`: data-only の単発ノイズ専用生成経路はない。
  - 根拠: `Dataset.__init_circuit`
  - 参照: `data.py:57-66`
- `phenomenological`: `未確認`
  - `理由`: repeated syndrome rounds と measurement error はあるが、Clifford/reset ノイズも同時に注入しているため、純粋 phenomenological 専用とは言えない。
- `circuit-level`: `対応`
  - `明示的記述`: 回路中の複数操作点にノイズを入れてサンプル/DEM を生成している。
  - 根拠: `Dataset.__init_circuit`, `test_mwpm`
  - 参照: `data.py:57-67`, `mwmp.py:14-16`

## 3. デコードアルゴリズムの概要

### 3.1 パイプライン一覧

#### パイプライン A: MWPM baseline

- `明示的記述`: `test_mwpm` は `Stim` 回路から detector error model を作り、`pymatching.Matching.from_detector_error_model(...)` で matcher を生成し、`matcher.decode_batch(detection_array)` を実行する。
  - 根拠: `test_mwpm`
  - 参照: `mwmp.py:9-25`
- `結論`: 採用アルゴリズムは MWPM である。
  - 根拠: `test_mwpm`
  - 参照: `mwmp.py:9-25`

#### パイプライン B: GNN + GRU neural decoder

- `明示的記述`: `GRUDecoder` は `GraphConvLayer` を複数段重ねた embedding 部、`GRU`、最後の `Linear + Sigmoid` から構成される。
  - 根拠: `GRUDecoder.__init__`, `GraphConvLayer`
  - 参照: `gru_decoder.py:15-31`, `utils.py:41-49`
- `明示的記述`: `forward` は `global_mean_pool` で chunk ごとのグラフ埋め込みを作り、`group(...)` で batch ごとの時系列に並べ、`GRU` の最終隠れ状態から 1 ビットの logical flip 確率を出す。
  - 根拠: `GRUDecoder.embed`, `GRUDecoder.forward`, `group`
  - 参照: `gru_decoder.py:33-42`, `utils.py:12-39`
- `結論`: 採用アルゴリズムは「Graph Neural Network による各 time chunk の埋め込み + Recurrent Neural Network による時系列統合」である。
  - 根拠: `GRUDecoder.__init__`, `GRUDecoder.forward`, `group`
  - 参照: `gru_decoder.py:15-42`, `utils.py:12-39`

### 3.2 `rotated_memory_z` / `rotated_memory_x` の扱い

- `明示的記述`: code family 切替は `Dataset.__init__` の `flip` 引数で行い、`FlipType.BIT -> surface_code:rotated_memory_z`、`FlipType.PHASE -> surface_code:rotated_memory_x` となる。
  - 根拠: `FlipType`, `Dataset.__init__`
  - 参照: `data.py:12-14`, `data.py:27-49`
- `明示的記述`: `GRUDecoder.train_model` は `Dataset(self.args)` をそのまま生成しており、引数未指定時は `FlipType.BIT` の既定値を使う。
  - 根拠: `Dataset.__init__` の既定引数, `GRUDecoder.train_model`
  - 参照: `data.py:27`, `gru_decoder.py:55-56`
- `推論`: neural decoder のアーキテクチャ自体は `rotated_memory_z` と `rotated_memory_x` で共有されるが、examples と標準学習経路は既定値の `BIT` 側に寄っている。`PHASE` 側は `Dataset(..., flip=FlipType.PHASE)` を明示的に使う追加コードが必要で、repo 付属スクリプトではその経路を実演していない。
  - 根拠: examples が `Dataset(args)` または `GRUDecoder(args)` のみを使うこと
  - 参照: `examples/train_nn.py:10-26`, `examples/test_nn.py:10-28`

### 3.3 物理誤りから syndrome graph への落とし込み

- `明示的記述`: repo が直接扱うのは「物理誤りそのもの」ではなく `Stim` sampler が返す detector events である。`sample_syndromes` の docstring も `Each entry indicates if there has been a detection event` と説明している。
  - 根拠: `Dataset.sample_syndromes`
  - 参照: `data.py:84-110`
- `未確認`: どの単一物理誤りがどの detector 集合に写るか、という fault-to-detector mapping を repo 独自に列挙・説明するコードはない。
  - 根拠: `Dataset` は `Stim` の detector 出力をそのまま受け取るだけで、物理 fault table を保持しない。
  - 参照: `data.py:98-110`, `mwmp.py:15-16`
- `推論`: したがって fault-to-syndrome の詳細は `Stim` 生成回路/DEM に委譲されており、repo から直接確認できるのは detector event 表現と stabilizer type のみである。

### 3.4 X/Z stabilizer の統合方法

- `明示的記述`: `Dataset.__init_circuit` は `syndrome_mask` を用いて X stabilizer 位置と Z stabilizer 位置を作り分け、`get_node_features` は stabilizer type を `(0, 1)=X`, `(1, 0)=Z` の 2 ビット特徴として node feature に追加する。
  - 根拠: `Dataset.__init_circuit`, `Dataset.get_node_features`
  - 参照: `data.py:76-82`, `data.py:169-200`
- `明示的記述`: `get_edges` は stabilizer type も含んだ 5 次元 node feature 上で `knn_graph` を構成し、type ごとに別グラフへ分離する処理は行わない。
  - 根拠: `Dataset.get_edges`, `Dataset.get_node_features`
  - 参照: `data.py:169-218`
- `結論`: neural decoder は X/Z を独立した matching 問題として明示分離せず、単一グラフ内で stabilizer type feature により区別している。
  - 根拠: `Dataset.get_node_features`, `Dataset.get_edges`
  - 参照: `data.py:169-218`
- `明示的記述`: MWPM 側も repo 内で X/Z 用の独立 matcher を手動構築せず、`Stim` の detector error model から単一の `Matching` を生成する。
  - 根拠: `test_mwpm`
  - 参照: `mwmp.py:14-16`
- `未確認`: `rotated_memory_z` と `rotated_memory_x` でどの物理 Pauli 成分が主にどの stabilizer type に検出されるかを repo は説明していない。

## 4. 入出力インターフェースとデコードの運用形態

### 4.1 MWPM パイプライン

- `入力データ`
  - `明示的記述`: `matcher.decode_batch` に渡すのは `detection_array` で、型は `bool` の NumPy 配列、shape は `[batch, num_detectors]`。
  - 根拠: `Dataset.sample_syndromes`, `test_mwpm`
  - 参照: `data.py:88-110`, `mwmp.py:21-23`
- `syndrome の定義`
  - `明示的記述`: 各ビットは stabilizer 値そのものではなく detector event である。
  - 根拠: `Dataset.sample_syndromes`
  - 参照: `data.py:88-93`
  - `未確認`: 最終ラウンドを data-qubit readout から repo 側で再構成する処理は見つからない。
  - `推論`: 必要な最終 detector 定義がある場合も、それは `Stim` の detector sampler 内部に隠れており、repo は生の data-qubit readout を扱わない。
- `出力データ`
  - `明示的記述`: 出力は `predictions` で、`flips_array` と比較される logical observable flip の予測である。
  - 根拠: `test_mwpm`, `Dataset.sample_syndromes`
  - 参照: `mwmp.py:21-25`, `data.py:94-110`
- `運用形態`
  - `結論`: `logical readout-only decoding`
  - `明示的記述`: 物理誤り列や補正操作列を返さず、logical flip ラベルとの一致率のみ評価している。
  - 根拠: `test_mwpm`
  - 参照: `mwmp.py:17-30`
  - `推論`: 1 ビットの logical frame 判定には転用できるが、repo には active correction として回路へ返す機構はない。

### 4.2 GNN + GRU パイプライン

- `入力データ`
  - `明示的記述`: `Dataset.generate_batch()` は以下を返す。
    - `node_features`: torch tensor `[n, 5]`
    - `edge_index`: torch tensor `[n_edges, 2]`
    - `labels`: torch tensor `[n]`
    - `label_map`: torch tensor `[n_graphs, 2]`
    - `edge_attr`: torch tensor `[n_edges]`
    - `flips`: torch tensor `[batch_size, 1]` 相当
  - 根拠: `Dataset.generate_batch`
  - 参照: `data.py:220-268`
- `node feature の意味`
  - `明示的記述`: 各 node は `[x, y, t, stabilizer_type_Z, stabilizer_type_X]` であり、detector event が立った stabilizer のみを node 化する。
  - 根拠: `Dataset.get_node_features`
  - 参照: `data.py:169-200`
- `syndrome の定義`
  - `明示的記述`: 入力元は detector event であり、生の stabilizer 測定値ではない。
  - 根拠: `Dataset.sample_syndromes`, `Dataset.get_node_features`
  - 参照: `data.py:84-110`, `data.py:182-199`
  - `未確認`: 最終ラウンドの syndrome を repo が data-qubit readout から再構成するコードはない。
- `グラフ化`
  - `明示的記述`: 各 chunk 内の node 群に対し `knn_graph` を用いて辺を張り、edge weight は node feature 差分ノルムの逆二乗である。
  - 根拠: `Dataset.get_edges`
  - 参照: `data.py:202-218`
- `時系列化`
  - `明示的記述`: `sliding=True` のとき `get_sliding_window` が time chunk を作り、`group` が batch ごとに可変長系列へパックし、`GRU` に渡す。
  - 根拠: `Dataset.get_sliding_window`, `utils.group`, `GRUDecoder.forward`
  - 参照: `data.py:112-167`, `utils.py:12-39`, `gru_decoder.py:38-42`
- `出力データ`
  - `明示的記述`: 出力は `Sigmoid` 後の単一確率で、教師信号 `flips` に対して `BCELoss` で学習する。
  - 根拠: `GRUDecoder.__init__`, `GRUDecoder.train_model`
  - 参照: `gru_decoder.py:22-25`, `gru_decoder.py:57-84`
- `運用形態`
  - `結論`: `logical readout-only decoding`
  - `明示的記述`: モデル出力は logical bit/phase flip の有無であり、物理 qubit ごとの correction string や matching edge 重みは返さない。
  - 根拠: `Dataset.sample_syndromes`, `GRUDecoder.forward`, `GRUDecoder.test_model`
  - 参照: `data.py:94-110`, `gru_decoder.py:38-42`, `gru_decoder.py:136-157`
  - `推論`: active correction や full correction ではなく、fault-tolerant memory/readout の論理成否判定用途に限定される。

### 4.3 full correction / Pauli frame update / logical readout-only の切り分け

- `full correction`: `非対応`
  - `明示的記述`: 物理 qubit 上の correction operator を出力する API がない。
  - 根拠: `GRUDecoder`, `test_mwpm`, `Dataset.sample_syndromes`
  - 参照: `gru_decoder.py:11-157`, `mwmp.py:9-30`, `data.py:94-110`
- `Pauli frame update`: `未確認`
  - `理由`: 出力は 1 ビットの logical flip なので、事後的な logical frame 更新に使う余地はあるが、repo 自体はその運用を実装・説明していない。
- `logical readout-only decoding`: `対応`
  - `明示的記述`: 全評価コードが `observable_flips` / `flips` との一致率のみを計測している。
  - 根拠: `Dataset.sample_syndromes`, `GRUDecoder.test_model`, `test_mwpm`, README 図の説明
  - 参照: `data.py:94-110`, `gru_decoder.py:136-157`, `mwmp.py:9-30`, `README.md:5-7`

## 5. Neural network 系アルゴリズムの対応

### 5.1 training / inference の対応有無

- `明示的記述`: inference には対応している。`GRUDecoder.test_model` が評価 API を持ち、`examples/test_nn.py` と `examples/load_nn.py` が学習済み重みの読み込みを示している。
  - 根拠: `GRUDecoder.test_model`, `examples/test_nn.py`, `examples/load_nn.py`
  - 参照: `gru_decoder.py:136-157`, `examples/test_nn.py:8-28`, `examples/load_nn.py:7-62`
- `明示的記述`: training にも対応している。`GRUDecoder.train_model` が学習ループを実装し、`examples/train_nn.py` が学習例を提供する。
  - 根拠: `GRUDecoder.train_model`, `examples/train_nn.py`
  - 参照: `gru_decoder.py:44-135`, `examples/train_nn.py:8-26`
- `明示的記述`: 学習済みモデルとして `models/distance3.pt`, `models/distance5.pt`, `models/distance7.pt` が同梱されている。
  - 根拠: ファイル実在, `examples/load_nn.py`
  - 参照: `examples/load_nn.py:23-24`, `examples/load_nn.py:42-43`, `examples/load_nn.py:61-62`

### 5.2 合成訓練データの生成方法

- `明示的記述`: 訓練データは静的ファイルから読むのではなく、`train_model` 内で `dataset = Dataset(self.args)` を作り、各 minibatch で `dataset.generate_batch()` を呼んでオンザフライ生成する。
  - 根拠: `GRUDecoder.train_model`
  - 参照: `gru_decoder.py:55-56`, `gru_decoder.py:74-84`
- `明示的記述`: `Dataset.__init_circuit` は `itertools.product(self.error_rates, self.t)` に対して `Stim` 回路群を用意し、`generate_batch` はその中から `sampler_idx = np.random.choice(len(self.samplers))` で 1 つ選んでサンプルする。
  - 根拠: `Dataset.__init_circuit`, `Dataset.generate_batch`
  - 参照: `data.py:55-67`, `data.py:243-258`
- `結論`: 合成訓練データは、指定された `error_rates` と `t` の直積にまたがる circuit-level surface-code memory 実験をオンライン生成したものである。
  - 根拠: `Dataset.__init_circuit`, `Dataset.generate_batch`, `GRUDecoder.train_model`
  - 参照: `data.py:55-67`, `data.py:243-258`, `gru_decoder.py:55-84`
- `明示的記述`: `sample_syndromes` は `compile_detector_sampler(...).sample(shots=..., separate_observables=True)` を使って detector events と logical observable flips を同時に取得する。
  - 根拠: `Dataset.sample_syndromes`
  - 参照: `data.py:98-110`
- `明示的記述`: 生成後、detector event が 1 つもない shot は除外される。コード中コメントも「only include cases where there is at least one detection event」と記す。
  - 根拠: `Dataset.sample_syndromes`
  - 参照: `data.py:100-109`
- `推論`: したがって NN は「少なくとも 1 つ detection event が起きた事例」に条件付けられたデータで学習される。
- `明示的記述`: detector event は node feature `[x, y, t, stabilizer_type_Z, stabilizer_type_X]` に変換され、`knn_graph` でグラフ化され、`sliding=True` なら time chunk 列に分割される。
  - 根拠: `Dataset.get_node_features`, `Dataset.get_edges`, `Dataset.get_sliding_window`
  - 参照: `data.py:112-218`
- `明示的記述`: 教師信号は `flips` で、`BCELoss` を用いて二値分類として学習する。
  - 根拠: `GRUDecoder.train_model`
  - 参照: `gru_decoder.py:57-84`
- `推論`: 既定学習経路は `Dataset(self.args)` を使うため `FlipType.BIT`、すなわち `surface_code:rotated_memory_z` に固定される。phase 側学習には `train_model` 側の追加改変または別呼び出しが必要である。
  - 根拠: `Dataset.__init__` の既定値, `GRUDecoder.train_model`
  - 参照: `data.py:27-49`, `gru_decoder.py:55-56`

## 6. ベンチマークの評価内容

### 6.1 repo 内で明示的に存在する評価

- `明示的記述`: README は benchmark 結果として `Logical accuracy` と `Logical failure rate` の図を掲載している。
  - 根拠: `README.md`（関数/クラス名なし）
  - 参照: `README.md:5-7`
- `明示的記述`: 実行可能な評価スクリプトとしては `examples/test_nn.py` があり、学習済み distance-5 NN と MWPM を比較評価する。
  - 根拠: `examples/test_nn.py`（関数/クラス名なし）
  - 参照: `examples/test_nn.py:8-28`
- `明示的記述`: `GRUDecoder.test_model` と `test_mwpm` が実際に計算する主指標は accuracy とその標準偏差である。
  - 根拠: `GRUDecoder.test_model`, `test_mwpm`
  - 参照: `gru_decoder.py:141-157`, `mwmp.py:17-30`
- `未確認`: README 図を生成した plotting スクリプトや failure-rate 集計スクリプトは repo 内に見つからない。
  - 根拠: ファイル一覧、repo 全体検索

### 6.2 何を評価しているか

- `結論`: 評価対象は `logical memory` 実験の最終 logical observable 判定であり、`1-shot readout` ではない。
  - `明示的記述`: `Stim` 回路生成で `rounds=t` を指定し、`sample_syndromes` のラベルは「syndrome の最後で測定した logical bit/phase flip」の有無である。
  - 根拠: `Dataset.__init_circuit`, `Dataset.sample_syndromes`
  - 参照: `data.py:57-66`, `data.py:94-110`
- `明示的記述`: repeated rounds は example 評価で `t=[99]` に設定されている。
  - 根拠: `examples/test_nn.py`
  - 参照: `examples/test_nn.py:10-21`
- `推論`: よって、fault-tolerant memory あるいは長時間メモリ保持中の論理成否判定に近い評価である。

### 6.3 評価の前提条件

- `ラウンド数`
  - `明示的記述`: `examples/test_nn.py` は `t=[99]`, `dt=2`, `sliding=True` で評価する。
  - 根拠: `examples/test_nn.py`
  - 参照: `examples/test_nn.py:10-21`
- `distance`
  - `明示的記述`: 同スクリプトは `distance=5` のモデルを読む。
  - 根拠: `examples/test_nn.py`
  - 参照: `examples/test_nn.py:10-24`
- `誤り率`
  - `明示的記述`: 同スクリプトは `error_rates=[0.001]` で評価する。
  - 根拠: `examples/test_nn.py`
  - 参照: `examples/test_nn.py:10-13`
- `乱数シード`
  - `明示的記述`: `seed=42` を設定する。
  - 根拠: `examples/test_nn.py`
  - 参照: `examples/test_nn.py:18-21`
- `ショット選別`
  - `明示的記述`: 0 detection-event shot は除外される。
  - 根拠: `Dataset.sample_syndromes`
  - 参照: `data.py:100-109`
  - `推論`: したがって accuracy は「全ショット平均」ではなく「少なくとも 1 detector が発火したショット」に条件付けた評価である。
- `評価対象の偏り`
  - `明示的記述`: `examples/test_nn.py` は `Dataset(args)` を使うだけで `flip` を指定しない。
  - 根拠: `examples/test_nn.py`, `Dataset.__init__`
  - 参照: `examples/test_nn.py:26-28`, `data.py:27-49`
  - `推論`: benchmark は既定の `FlipType.BIT`、すなわち `surface_code:rotated_memory_z` 側に偏っており、`surface_code:rotated_memory_x` 側の benchmark は repo 付属スクリプトでは未提示である。
- `学習条件とのずれ`
  - `明示的記述`: `examples/load_nn.py` のコメントでは distance-5 と distance-7 モデルは `t = 49, dt = 2` で学習されたと説明される。
  - 根拠: `examples/load_nn.py`
  - 参照: `examples/load_nn.py:26-29`, `examples/load_nn.py:45-48`
  - `明示的記述`: 一方 `examples/test_nn.py` の distance-5 評価は `t=[99], dt=2` である。
  - 根拠: `examples/test_nn.py`
  - 参照: `examples/test_nn.py:10-15`
  - `推論`: 少なくとも付属の distance-5 評価例は、コメント上の学習時系列長より長い系列で推論している。
- `誤りモデル`
  - `明示的記述`: 評価は `after_clifford_depolarization`, `before_round_data_depolarization`, `before_measure_flip_probability`, `after_reset_flip_probability` を同一確率で注入する circuit-level ノイズを前提とする。
  - 根拠: `Dataset.__init_circuit`
  - 参照: `data.py:57-66`
- `logical Z 限定などの偏り`
  - `未確認`: repo は logical-Z のみ、logical-X のみ、あるいは bit/phase のどちらを figure に描いたかを README 図では明示していない。
  - `推論`: 付属 scripts と既定経路は `BIT -> rotated_memory_z` 側なので、少なくとも再現可能な benchmark 例はその側に偏る。

### 6.4 しきい値図・大域評価の有無

- `未確認`: threshold curve や threshold fitting を行うスクリプトは repo 内に見つからない。
- `未確認`: README 図ファイル名 `performance_t_accuracy.png` / `performance_t_failure.png` からは、横軸が `t` である可能性はあるが、repo はその生成条件を記録していない。
