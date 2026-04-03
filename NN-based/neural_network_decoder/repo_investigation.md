# Repo Investigation for `origin/arXiv1705p07855`

## 調査スコープ

- 明示的記述:
  - 現在の checkout は `origin/arXiv1705p07855` を `git switch --detach` した detached HEAD。根拠: `git status --short --branch`。関数/クラス名: 該当なし。
  - この branch 上で確認できる repo 内ファイルは `README.md` と `decoder.py`。根拠: `rg --files`。関数/クラス名: 該当なし。
- 推論:
  - 本メモはこの branch に存在する `README.md` と `decoder.py` のみを根拠に記述する。別 branch の notebook や `src/` 配下は根拠に含めない。

## 1. Surface code の対応状況と実装範囲

### 1-1. この branch は surface code の QEC に対応しているか

- 結論:
  - 対応している。
  - 対象実装は branch 直下の `decoder.py`。

- 明示的記述:
  - `README.md` は「decoder.py describes a decoder for stabilizer codes, in particular for the surface code」と明記する。根拠: `README.md`。関数/クラス名: 該当なし。
  - `decoder.py` のモジュールドキュメントは「in particular for the surface code」と記述する。根拠: `decoder.py`。関数/クラス名: 該当なし。
  - `Decoder` クラス docstring は「designed for quantum error correction on surface codes」と記述する。根拠: `decoder.py`。関数/クラス名: `Decoder`。

### 1-2. 対応するコードの種類

- 明示的記述:
  - repo 内で明示されるコード family は surface code。根拠: `README.md`, `decoder.py`。関数/クラス名: `Decoder`。
  - `__main__` の設定例は `dim_syndr=8`, `dim_fsyndr=4`。根拠: `decoder.py`。関数/クラス名: `Decoder`, `__main__`。
  - `Decoder._init_data_params` は `n_data_qubits = dim_syndr + 1`、`n_qubits = 2 * dim_syndr + 1` を設定する。根拠: `decoder.py`。関数/クラス名: `Decoder._init_data_params`。

- 推論:
  - 例示設定は `dim_syndr=8` から `n_qubits=17` となるため、小型の surface-17 系を想定している可能性が高い。
  - ただし、rotated planar か standard planar か、XXZZ か XZZX か、toric か planar かは repo 内に明示がなく未確認。

### 1-3. 実装制約

- 明示的記述:
  - README と `decoder.py` は単一 logical qubit 用であることを明示しないが、surface code decoder の用途説明は単一 parity 推定に集約されている。根拠: `README.md`, `decoder.py`。関数/クラス名: `Decoder._init_network_variables`, `Decoder.calc_fids_and_plog`, `Decoder.benchmark`。
  - 出力ラベルは `parity` 1 ビットである。根拠: `decoder.py`。関数/クラス名: `Decoder._init_network_variables`, `Decoder.gen_batch`, `Decoder.gen_batch_oversample`。

- 推論:
  - 実装上の出力が単一 parity であるため、multi-logical-qubit patch や複数 logical observable を同時に扱う設計ではない。
  - odd distance 限定、open boundary 限定、rectangular patch 可否については repo 内に証拠がなく未確認。

### 1-4. Capability Matrix

| code family | patch shape | single/multi logical qubit | boundaries | odd distance restriction | repeated syndrome rounds | measurement error support | active correction support | lattice surgery | benchmark scripts | neural decoder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| surface code | 未確認。surface code とは明示されるが rotated planar / standard planar / toric / XXZZ / XZZX は未確認。根拠: `README.md`, `decoder.py` `Decoder` | single 相当。出力が単一 parity であるため。根拠: `decoder.py` `Decoder._init_network_variables`, `Decoder.calc_fids_and_plog` | 未確認 | 未確認 | 対応あり。時系列 `events` と `length` を入力し、複数 cycle の fidelity decay を評価。根拠: `decoder.py` `Decoder.gen_batch`, `Decoder.gen_batch_oversample`, `Decoder.network`, `Decoder.benchmark` | 未確認。`events` は扱うが、それが measurement error を含む detection event かは repo 内で未確認。根拠: `decoder.py` `Decoder.gen_batch`, `Decoder.gen_batch_oversample` | 非対応寄り。物理補正列ではなく parity 予測のみを返す。根拠: `decoder.py` `Decoder._init_network_variables`, `Decoder.calc_fids_and_plog`, `Decoder.benchmark` | 証拠なし。加えて出力設計も parity 推定専用。根拠: `decoder.py` `Decoder` | あり。`benchmark` と `__main__` がある。根拠: `decoder.py` `Decoder.benchmark`, `__main__` | あり。LSTM + feedforward。根拠: `decoder.py` `Decoder.network`, `Decoder.network_fsyndr` |

### 1-5. 未対応理由の切り分け

- lattice surgery:
  - 明示的記述: 関連 API やデータ構造は repo 内に見当たらない。根拠: `README.md`, `decoder.py`。関数/クラス名: `Decoder`。
  - 推論: 単なる未実装に加え、出力が単一 parity 予測に固定されているため、現在の前提アーキテクチャ自体が lattice surgery には不向き。

- active correction:
  - 明示的記述: 返すのは parity 推定と benchmark 統計であり、補正 Pauli 列ではない。根拠: `decoder.py`。関数/クラス名: `Decoder.calc_fids_and_plog`, `Decoder.benchmark`, `calc_stats`。
  - 推論: 単に callback がないだけでなく、出力表現が active correction 用になっていない。

## 2. 対象ノイズモデル

### 2-1. decoder 実装が仮定するノイズモデル

- 明示的記述:
  - network 入力は `events`, `err_signal`, `parity`, `length` であり、ノイズ率、エラーチャネル、matching weight を内部で計算しない。根拠: `decoder.py`。関数/クラス名: `Decoder._init_network_variables`, `Decoder.gen_batch`, `Decoder.gen_batches`, `Decoder.calc_fids_and_plog`。
  - `Decoder._init_network_functions` は 2 つの neural network 出力から parity の sigmoid を作る。根拠: `decoder.py`。関数/クラス名: `Decoder._init_network_functions`。

- 推論:
  - decoder ロジック自体は code capacity / phenomenological / circuit-level のいずれにも固定されていない。
  - ノイズモデル依存性は学習・評価用 SQLite database に押し込まれている。
  - measurement error, データ誤りのみ, 相関誤り, Y 誤り, biased noise のいずれについても、decoder 内に専用分岐や明示的な重み計算は存在しない。

### 2-2. benchmark / simulation スクリプトが実際に使うノイズモデル

- 明示的記述:
  - README はこの branch が Ref. [2] を説明するとし、その reference title は「Machine-learning-assisted correction of correlated qubit errors in a topological code」。根拠: `README.md`。関数/クラス名: 該当なし。
  - `decoder.py` は benchmark を実装するが、ノイズ生成シミュレータは含まない。benchmark は既存 DB を読む。根拠: `decoder.py`。関数/クラス名: `Decoder.load_data`, `Decoder.gen_batches`, `Decoder.benchmark`。
  - `__main__` は DB パスとファイル名をユーザ設定に委ねている。根拠: `decoder.py`。関数/クラス名: `__main__`。

- 推論:
  - repo 内 benchmark スクリプト自身は、実際のノイズモデルを定義しない。
  - この branch が想定する実験文脈は README 上は correlated qubit errors だが、SQLite DB の中身を生成するコードがないため、code capacity / phenomenological / circuit-level のどれかは repo 内だけでは確定できない。

### 2-3. 要素別整理

| 観点 | 調査結果 |
| --- | --- |
| decoder 実装が固定する前提モデル | 未固定。入力済み `events` / `err_signal` を parity 分類に使うだけ。根拠: `decoder.py` `Decoder._init_network_variables`, `Decoder._init_network_functions` |
| benchmark が実際に使う前提モデル | DB 依存。repo 内では未確認。README の文脈上は correlated qubit errors。根拠: `README.md`, `decoder.py` `Decoder.load_data`, `Decoder.benchmark` |
| measurement error | 未確認 |
| データ誤りのみか | 未確認 |
| 相関誤り | README の reference title 上は対象。実装側に専用ロジックはない。根拠: `README.md`, `decoder.py` `Decoder` |
| Y 誤り | 未確認 |
| biased noise | 未確認 |

## 3. デコードアルゴリズムの概要

### 3-1. パイプライン

- 明示的記述:
  - `Decoder.network` は full history `x1` を処理する LSTM + feedforward。根拠: `decoder.py`。関数/クラス名: `Decoder.network`。
  - `Decoder.network_fsyndr` は末尾 `n_steps_net2` の `x2` と `final syndrome increment` `fx` を処理する別の LSTM + feedforward。根拠: `decoder.py`。関数/クラス名: `Decoder.network_fsyndr`。
  - `_init_network_functions` は両者の出力 `out1`, `out2` を `reduce_logsumexp` を用いて結合し、`self.predictions` を作る。根拠: `decoder.py`。関数/クラス名: `Decoder._init_network_functions`。
  - 学習は `sigmoid_cross_entropy` による parity 二値分類。根拠: `decoder.py`。関数/クラス名: `Decoder._init_network_functions`。

- 推論:
  - 採用アルゴリズムは recurrent neural network decoder であり、MWPM, CNN, GNN は実装されていない。

### 3-2. surface-code variant の扱い

- 明示的記述:
  - repo 内に syndrome graph, matching graph, X/Z 独立 matching, variant 別分岐は存在しない。根拠: `README.md`, `decoder.py`。関数/クラス名: `Decoder`。

- 推論:
  - XXZZ と XZZX のような variant を共通 graph decoder で扱う実装ではない。
  - 物理誤りがどの syndrome graph に落ちるか、独立な X/Z matching と見なすかは、repo 内コードでは未確認。
  - もし variant 差があっても、それは外部前処理済み `events` / `err_signal` の意味付けに埋め込まれていると推測される。

## 4. 入出力インターフェースとデコードの運用形態

### 4-1. 入力データ

- 明示的記述:
  - `Decoder.gen_batches` は training/validation/test DB から `events`, `err_signal`, `parity`, `length` または oversample 用に `events`, `err_signal`, `parities` を読む。根拠: `decoder.py`。関数/クラス名: `Decoder.gen_batches`。
  - `Decoder.gen_batch` は 1 サンプルを `syndr1`, `syndr2`, `fsyndr`, `len1`, `len2`, `parity` に整形する。根拠: `decoder.py`。関数/クラス名: `Decoder.gen_batch`。
  - placeholder は `x1`, `x2`, `fx`, `y`, `l1`, `l2`。根拠: `decoder.py`。関数/クラス名: `Decoder._init_network_variables`。

- 推論:
  - 入力は graph ではなく tensor。
  - `x1` は full history、`x2` は末尾 window、`fx` は最終 readout 由来の補助情報。

### 4-2. Syndrome の定義

- 明示的記述:
  - 入力ベクトルは docstring 上 `syndrome increments (events)` と呼ばれる。根拠: `decoder.py`。関数/クラス名: `Decoder`。
  - `dim_fsyndr` は `final syndrome increments (error signal)` の次元であり、最終 data-qubit readout 由来の情報と説明される。根拠: `decoder.py`。関数/クラス名: `Decoder`。
  - `gen_batch_oversample` は、各 time step の final syndrome increment を複数サンプルに分解する。根拠: `decoder.py`。関数/クラス名: `Decoder.gen_batch_oversample`。

- 推論:
  - 各ビットは stabilizer 値そのものではなく、ラウンド差分ないし detection event 相当。
  - 最終ラウンド情報は data-qubit readout から再構成された `err_signal` / `final syndrome increment` として別入力に供給される。

### 4-3. 出力データ

- 明示的記述:
  - 出力は `parity of bitflips` の確率。根拠: `decoder.py`。関数/クラス名: `Decoder._init_network_variables`, `Decoder._init_network_functions`。
  - `calc_fids_and_plog` と `benchmark` はその確率を 0.5 閾値で 0/1 にし、真値 parity と比較する。根拠: `decoder.py`。関数/クラス名: `Decoder.calc_fids_and_plog`, `Decoder.benchmark`。

- 推論:
  - 物理エラー string は出力しない。
  - MWPM 用エッジ重みは出力しない。
  - 出力は logical observable parity の直接予測。

### 4-4. 運用形態

- 明示的記述:
  - benchmark 出力は fidelity / logical error rate の統計辞書。根拠: `README.md`, `decoder.py`。関数/クラス名: `Decoder.benchmark`, `calc_stats`。
  - 補正パターンや Pauli frame を返す API は存在しない。根拠: `decoder.py`。関数/クラス名: `Decoder`。

- 推論:
  - full correction ではない。
  - Pauli frame update を明示的に返す実装でもない。
  - logical readout-only decoding に分類するのが最も近い。
  - active correction 非対応は、単なる未実装に加えて、出力表現が parity 1 ビットに固定されている点が構造的制約。

## 5. Neural network 系アルゴリズムの対応

### 5-1. training / inference 対応

- 明示的記述:
  - training は `Decoder.train_one_epoch` と `Decoder.do_training` が担う。根拠: `decoder.py`。関数/クラス名: `Decoder.train_one_epoch`, `Decoder.do_training`。
  - inference / evaluation は `Decoder.calc_fidelity`, `Decoder.calc_fids_and_plog`, `Decoder.benchmark` が担う。根拠: `decoder.py`。関数/クラス名: `Decoder.calc_fidelity`, `Decoder.calc_fids_and_plog`, `Decoder.benchmark`。

- 推論:
  - training と inference の両方に対応している。

### 5-2. 合成訓練データ生成

- 明示的記述:
  - repo 内には SQLite DB を生成するコードが存在しない。読み込みのみ。根拠: `README.md`, `decoder.py`。関数/クラス名: `Decoder.load_data`, `Decoder.gen_batches`。
  - `__main__` は既存 DB のパスを与える前提。根拠: `decoder.py`。関数/クラス名: `__main__`。

- 推論:
  - 合成訓練データ生成はこの repo の担当外。
  - 未対応理由は主として未実装。README の運用説明から見ると、データ生成は外部プロジェクトまたは別手順に委ねられている。

## 6. ベンチマークの評価内容

### 6-1. 何を評価しているか

- 明示的記述:
  - `benchmark` は test data 上で prediction と parity ラベルを比較し、`calc_stats` により fidelity decay と logical error rate `plog` を計算する。根拠: `decoder.py`。関数/クラス名: `Decoder.benchmark`, `calc_stats`。
  - `decay` / `calc_stats` は cycles に対する fidelity decay を指数関数で fit する。根拠: `decoder.py`。関数/クラス名: `decay`, `calc_stats`。

- 推論:
  - 評価内容は 1-shot readout ではなく、repeated syndrome rounds 上の logical memory 評価。

### 6-2. 評価前提条件

- 明示的記述:
  - `benchmark` は長い test sequence に合わせて `change_network_length` で network 長を変更してから評価する。根拠: `decoder.py`。関数/クラス名: `Decoder.benchmark`, `Decoder.change_network_length`。
  - `benchmark(... oversample=True)` は、各 time step に final syndrome increment がある run を各 cycle の単一サンプルへ分解して評価する。根拠: `decoder.py`。関数/クラス名: `Decoder.benchmark`, `Decoder.gen_batch_oversample`。
  - `__main__` の例では `oversample=True`, `max_steps=300`, `N_max=5 * 10**4`, `N_batches=100`。根拠: `decoder.py`。関数/クラス名: `__main__`。

- 推論:
  - 評価は各 cycle における parity 正答率から logical decay を作る memory 実験。
  - 評価対象は単一 parity label のため、logical Z 限定か measurement basis parity 限定かは示唆されるが、X/Z のどちらかは repo 内で未確認。
  - しきい値図がどの誤りモデルを前提にするかは、外部 DB 生成過程がないため未確認。

## 総括

- 明示的記述:
  - この branch は surface code 向け neural decoder branch であり、主実装は `decoder.py` のみ。根拠: `README.md`, `decoder.py`。関数/クラス名: `Decoder`。
  - 実装が直接扱うのは `events` と `final syndrome increment` からの parity 推定。根拠: `decoder.py`。関数/クラス名: `Decoder.gen_batch`, `Decoder.network`, `Decoder.network_fsyndr`, `Decoder.calc_fids_and_plog`。
- 推論:
  - この branch は surface code 用 decoder ではあるが、surface-code 幾何や variant を明示的にコード化した repo ではない。
  - 実体としては「外部で前処理・生成された dataset を受けて、論理 parity を直接予測する recurrent neural decoder」である。
