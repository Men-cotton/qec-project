# Repo Investigation: Surface-Code Related Support in `graphqec-paper`

更新方針:
- このファイルは調査段階ごとに追記する。
- 各主張について、根拠ファイルと関数/クラス名を併記する。
- `明示的記述` と `推論` を分離する。
- 根拠がない場合は `未確認` と書く。

## 1. Surface code の対応状況と実装範囲

### 1.1 結論

- `明示的記述`: この repo は surface code に一部対応している。README の supported code families に `Sycamore Surface Codes` が列挙されている。
  - 根拠: `README.md`, セクション `Scope`
- `明示的記述`: 実コードとして読み込まれる surface code 実装は `graphqec/qecc/surface_code/google_block_memory.py` の `SycamoreSurfaceCode` である。
  - 根拠: `graphqec/qecc/__init__.py`, `try: from .surface_code.google_block_memory import *`; `graphqec/qecc/surface_code/google_block_memory.py`, クラス `SycamoreSurfaceCode`
- `明示的記述`: `graphqec/qecc/__init__.py` は `RotatedSurfaceCode` と `ZuchongzhiSurfaceCode` を条件付き import するが、対応ファイル `stim_block_memory.py` / `ustc_block_memory.py` は repo 内に存在しない。
  - 根拠: `graphqec/qecc/__init__.py`, `try` import 節; repo 全体のファイル一覧
- `推論`: この repo 内で実際に調査可能な surface code 実装範囲は、Google/Sycamore 実験データを前提にした `SycamoreSurfaceCode` に限られる。generic な rotated planar surface code 実装は、少なくともこの checkout には同梱されていない。
  - 根拠: `graphqec/qecc/__init__.py`, 関数 `get_code`; `graphqec/qecc/surface_code/google_block_memory.py`, クラス `SycamoreSurfaceCode`

### 1.2 対象ディレクトリ

- `graphqec/qecc/surface_code/`
  - `google_block_memory.py`
  - `google_utils.py`
- `graphqec/benchmark/evaluate.py`
  - `benchmark_sycamore_acc`
- `graphqec/benchmark/evaluate_sycamore.py`
  - Sycamore 実験評価の旧系エントリポイント
- `graphqec/decoder/*`
  - `PyMatching`, `BPOSD`, `SlidingWindowBPOSD`, neural decoder 群はいずれも `SycamoreSurfaceCode.get_dem()` または `get_tanner_graph()` を通じて surface code に接続される

### 1.3 どの surface code か

- `明示的記述`: 実装名と README 上の名称は `Sycamore Surface Codes` / `SycamoreSurfaceCode` である。
  - 根拠: `README.md`, セクション `Scope`; `graphqec/qecc/surface_code/google_block_memory.py`, クラス `SycamoreSurfaceCode`
- `明示的記述`: profile は `Gd3X_N/E/S/W`, `Gd3Z_N/E/S/W`, `Gd5X`, `Gd5Z` に限定されている。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, クラス属性 `_PROFILES`
- `明示的記述`: 実装は `basis` を `X` または `Z` として扱い、実験データのパスも `surface_code_b{basis}_d{distance}_...` の形式で選択する。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, クラス `QECDataGoogleFormat.__init__`
- `明示的記述`: `remove_irrelevent_detectors` の docstring は `since it is a CSS code` としており、実装も detector を半分だけ保持する前提を置いている。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `remove_irrelevent_detectors`
- `推論`: repo 内根拠から確認できるのは「Sycamore の CSS 系 surface code」であり、`rotated planar XXZZ` か `XZZX` かを repo 内の明示記述だけで断定することはできない。少なくとも `XZZX` を明示するコード・docstring・config は見当たらない。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `remove_irrelevent_detectors`; `README.md`, セクション `Scope`; repo 全体検索結果

### 1.4 実装上の制約

- `明示的記述`: 実験データソースは `google` 固定である。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, クラス `QECDataGoogleFormat.__init__`
- `明示的記述`: `QEC_DATA_PATH` 環境変数が必須で、外部 Sycamore データが同梱されていない。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, module top-level; `README.md`, セクション `Assets`; `scripts/check_reproducibility.py`, 関数 `validate_config`
- `明示的記述`: `get_exp_data` は odd cycle のみを許し、`assert (num_cycle+1) % 2 == 1` を持つ。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.get_exp_data`
- `明示的記述`: 初期化時にロードする実データは `for r in range(1,26,2)` で、1,3,...,25 cycle のみ。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.__init__`
- `明示的記述`: parity 分岐付きの incremental circuit は `num_cycle <= 24` に制限される。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.get_syndrome_circuit`; 関数 `SycamoreSurfaceCode.get_exp_data`
- `推論`: benchmark/運用上の repeated syndrome rounds は 25 round までの odd-round memory experiment に実質制限されている。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.__init__`, `SycamoreSurfaceCode.get_exp_data`
- `推論`: multi-logical qubit は対象外である可能性が高い。`_get_tanner_graph` は `data_to_logical` を `(data_idx, 0)` のみで構築しており、複数 logical observable へ一般化していない。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode._get_tanner_graph`
- `推論`: rectangular patch 対応の証拠はない。`distance` は単一スカラーで、profile も `d=3,5` のみである。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, クラス属性 `_PROFILES`; 関数 `SycamoreSurfaceCode.__init__`
- `推論`: even distance 一般対応の証拠はない。constructor は `distance` を受けるが、同梱 profile は odd の 3 と 5 のみで、外部データ依存のため追加距離を repo 単体では確認できない。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, クラス属性 `_PROFILES`; 関数 `QECDataGoogleFormat.__init__`
- `未確認`: open boundary / closed boundary の別を repo 内 docstring や README が明示していない。
- `未確認`: `rotated planar XXZZ` と明示する記述は repo 内で未発見。
- `未確認`: lattice surgery, patch merge/split, multi-patch routing に関する API は未発見。

### 1.5 Capability Matrix

| code family | patch shape | single/multi logical qubit | boundaries | odd distance restriction | repeated syndrome rounds | measurement error support | active correction support | lattice surgery | benchmark scripts | neural decoder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `SycamoreSurfaceCode` (`basis=X/Z`, Google experimental dataset-backed CSS surface code) | `明示的記述`: 単一 `distance` 指定の patch。`推論`: square patch を示唆。rectangular は証拠なし | `推論`: single logical qubit 前提が強い。multi-logical の一般化なし | `未確認` | `明示的記述`: bundled profile は `d=3,5` のみ。`推論`: odd-distance 実用限定 | `明示的記述`: odd rounds 1..25 の実験データ。incremental circuit は parity 付きで `<=24` | `明示的記述`: あり。detector error model と detection events を扱う | `明示的記述`: decoder 出力は observable flip 予測。`推論`: active correction feed-back は未対応 | `未確認` ではなく `設計上対象外の証拠が濃い`: 対応 API/回路操作が存在しない | `明示的記述`: `graphqec/benchmark/evaluate.py` に `benchmark_sycamore_acc`、`graphqec/benchmark/evaluate_sycamore.py` に旧 benchmark | `明示的記述`: `GraphRNNDecoderV5A`, `GraphLinearAttnDecoderV2A` を `build_neural_decoder` で接続可能 |

### 1.6 未対応理由の切り分け

- generic surface code 実装
  - 判定: `未実装ではなく、この checkout では同梱されていないため未確認`
  - 根拠: `graphqec/qecc/__init__.py` は `RotatedSurfaceCode` / `ZuchongzhiSurfaceCode` を条件付き import するが、対応ファイル自体が repo 内にない
- lattice surgery
  - 判定: `設計上対象外の可能性が高い`
  - 根拠: `SycamoreSurfaceCode` の API は memory experiment 相当の `get_syndrome_circuit`, `get_dem`, `get_exp_data`, `get_tanner_graph` に閉じており、patch merge/split や複数 logical patch を表す API がない
- rectangular patch / arbitrary geometry
  - 判定: `未実装の可能性が高い`
  - 根拠: `distance` 単独指定と固定 profile のみで、長方形サイズや境界種別を指定する引数が存在しない

## 2. 対象ノイズモデル

### 2.1 デコーダ実装が仮定するノイズモデル

#### 2.1.1 DEM ベース classical decoder (`BPOSD`, `PyMatching`, `SlidingWindowBPOSD`)

- `明示的記述`: いずれも `test_code.get_dem(num_cycle, ...)` を入力に初期化される。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `_get_decoder`
- `明示的記述`: `BPOSD` は DEM を `detector_graph`, `obs_graph`, `priors` に変換し、BP+OSD を detector graph 上で実行する。
  - 根拠: `graphqec/decoder/bposd.py`, クラス `BPOSD.__init__`, `BPOSD.get_result`; `graphqec/qecc/utils.py`, 関数 `dem_to_detector_graph`
- `明示的記述`: `SlidingWindowBPOSD` も DEM を detector graph に変換し、detector 行列を時系列 window に分割して `BpOsdDecoder` を回す。
  - 根拠: `graphqec/decoder/_slidingwindow_bposd.py`, クラス `SlidingWindowBPOSD.__init__`, `SlidingWindowBPOSD.decode`
- `明示的記述`: `PyMatching` は DEM をそのまま `Matching.from_detector_error_model(dem)` に渡す。
  - 根拠: `graphqec/decoder/pymatching.py`, 関数 `_process_batch`
- `推論`: これら classical decoder の前提は code-capacity ではなく、`detector` と `observable` を含む DEM 上の fault-event モデルである。特に repeated syndrome rounds と measurement error を含む detector history を入力にしているため、repo 内 surface-code path では phenomenological より circuit-level / detector-level に近い。
  - 根拠: `graphqec/qecc/utils.py`, 関数 `dem_to_detector_graph`; `graphqec/benchmark/evaluate.py`, 関数 `benchmark_batch_acc`; `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.get_dem`
- `明示的記述`: `SlidingWindowBPOSD` は `num_detectors_per_cycle` を必要とし、detector 行列を half-cycle 単位で window 化する。
  - 根拠: `graphqec/decoder/_slidingwindow_bposd.py`, クラス `SlidingWindowBPOSD.__init__`, `_initialize_regions_and_windows`
- `推論`: `SlidingWindowBPOSD` は repeated syndrome rounds を強く前提としており、surface-code memory 実験向けの時系列 detector 構造を仮定している。
  - 根拠: `graphqec/decoder/_slidingwindow_bposd.py`, クラス `SlidingWindowBPOSD`

#### 2.1.2 Neural decoder (`GraphRNNDecoderV5A`, `GraphLinearAttnDecoderV2A`)

- `明示的記述`: neural decoder は解析的なノイズ重み計算を持たず、raw syndrome を `(encoding_syndromes, cycle_syndromes, readout_syndromes)` に分けて学習済みモデルへ通す。
  - 根拠: `graphqec/decoder/nn/models.py`, クラス `QECCDecoder._decode`; クラス `GraphRNNDecoderV5._simple_forward`; クラス `GraphLinearAttnDecoderV2._simple_forward`
- `推論`: neural decoder のロジック自体は code-capacity / phenomenological / circuit-level のいずれにも固定されていない。前提ノイズモデルは、学習時・推論時に与えられるデータ分布に依存する。
  - 根拠: `graphqec/decoder/nn/models.py`, クラス `QECCDecoder`; `graphqec/decoder/nn/train_utils.py`, 関数 `get_dataloaders`

#### 2.1.3 measurement error / 相関誤り / Y 誤り / biased noise

- `明示的記述`: measurement error は detector history に含まれる。`get_exp_data` は `detection_events.b8` を読み、benchmark でも `sampler.sample()` の syndrome を decoder に渡す。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `QECDataGoogleFormat.get_events`, `SycamoreSurfaceCode.get_exp_data`; `graphqec/benchmark/evaluate.py`, 関数 `benchmark_batch_acc`
- `明示的記述`: `reverse_engineer_stim_circuit_then_split_into_two_parts` は DEM から representative error を選び、1 体または 2 体の generic `E` fault として circuit に戻す。
  - 根拠: `graphqec/qecc/surface_code/google_utils.py`, 関数 `reverse_engineer_stim_circuit_then_split_into_two_parts`; 関数 `_replace_errors_with_representation`
- `推論`: repo は single-qubit X/Z のみには限定されていない。少なくとも DEM から復元される 1 体・2 体の generic Pauli fault を扱えるため、Y 誤りや 2 体相関誤りを排除していない。
  - 根拠: `graphqec/qecc/surface_code/google_utils.py`, 関数 `_replace_errors_with_representation`
- `明示的記述`: `BPOSD` は identical hyper-edge を `simplify_dem` でまとめるが、hyper-edge 自体は detector/observable の集合として保持される。
  - 根拠: `graphqec/decoder/bposd.py`, クラス `BPOSD.__init__`; `graphqec/qecc/utils.py`, 関数 `simplify_dem`
- `推論`: BPOSD / SlidingWindowBPOSD は graph-like pair-edge 専用ではなく、DEM の列として表現された相関 detector event を扱う。ただし `PyMatching` については repo 側で hyper-edge 分解をしていないため、対応範囲は `Matching.from_detector_error_model` 任せである。
  - 根拠: `graphqec/decoder/bposd.py`, クラス `BPOSD`; `graphqec/decoder/pymatching.py`, 関数 `_process_batch`
- `明示的記述`: surface-code path に biased-noise 用のパラメータや重み付け分岐はない。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, クラス `SycamoreSurfaceCode`; `graphqec/benchmark/evaluate.py`, 関数 `_get_decoder`
- `未確認`: external Google DEM に実際どの程度の高次相関誤りが含まれているかは、この repo 単体では確認できない。

### 2.2 ベンチマーク / シミュレーションスクリプトが実際に使うノイズモデル

#### 2.2.1 surface code 実験 benchmark (`sycamore_acc`)

- `明示的記述`: `benchmark_sycamore_acc` は `test_code.get_exp_data(r - 1, parity=parity)` を直接読み、実験由来の detection events と observable flips を評価する。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `benchmark_sycamore_acc`
- `推論`: この benchmark は code-capacity / phenomenological / circuit-level の synthetic model ではなく、実験データ評価である。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `QECDataGoogleFormat.get_events`, `QECDataGoogleFormat.get_obs_flips`; `README.md`, セクション `Assets`

#### 2.2.2 surface code の synthetic simulation / dataloader

- `明示的記述`: `IncrementalSimDataset` と `SimDataset` は `stim.FlipSimulator` を `code.get_syndrome_circuit(...)` に対して実行する。
  - 根拠: `graphqec/decoder/nn/dataloader.py`, クラス `IncrementalSimDataset.__getitem__`, `SimDataset.__getitem__`
- `明示的記述`: `SycamoreSurfaceCode.get_syndrome_circuit` は、Google 実データ由来の noisy circuit / DEM から作られた incremental circuit を返す。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, クラス `SycamoreSurfaceCode.__init__`; 関数 `SycamoreSurfaceCode.get_syndrome_circuit`; 関数 `build_incremental_circuits`
- `推論`: repo 内の surface-code synthetic simulation は circuit-level に近い detector-event model を使っている。理由は、noisy circuit と DEM を元に `stim` circuit を再構成して `FlipSimulator` でサンプルしているため。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, クラス `SycamoreSurfaceCode`; `graphqec/qecc/surface_code/google_utils.py`, 関数 `reverse_engineer_stim_circuit_then_split_into_two_parts`

#### 2.2.3 `physical_error_rate` 引数の扱い

- `明示的記述`: benchmark 共通コードは `test_code.get_dem(num_cycle, physical_error_rate=error_rate)` を呼ぶ。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `_get_decoder`, `benchmark_batch_acc`, `benchmark_batch_time`
- `明示的記述`: `SycamoreSurfaceCode.get_dem` / `get_syndrome_circuit` は `**kwargs` を受けるが、`physical_error_rate` を使わない。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.get_dem`, `SycamoreSurfaceCode.get_syndrome_circuit`
- `推論`: generic `acc` / `time` benchmark の `error_rate` sweep は surface-code path では実効的にノイズ強度を変えない。surface code で意味のある評価経路は repo 上は `sycamore_acc` または dataloader の `exp` / `sim` path である。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `_get_decoder`, `benchmark_batch_acc`, `benchmark_batch_time`; `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.get_dem`

## 3. デコードアルゴリズムの概要

### 3.1 DEM ベース classical pipeline

#### 3.1.1 BPOSD

- `明示的記述`: パイプラインは `DEM -> detector_graph/obs_graph/priors -> BpOsdDecoder -> estimated fault columns -> obs_graph との積 -> logical flip prediction` である。
  - 根拠: `graphqec/decoder/bposd.py`, クラス `BPOSD.__init__`, `_process_batch`, `get_result`
- `明示的記述`: public API の返り値は物理誤り列ではなく logical observable prediction である。
  - 根拠: `graphqec/decoder/bposd.py`, 関数 `BPOSD.get_result`

#### 3.1.2 PyMatching

- `明示的記述`: パイプラインは `DEM -> Matching.from_detector_error_model(dem) -> decode_batch(raw_syndromes)` である。
  - 根拠: `graphqec/decoder/pymatching.py`, 関数 `_process_batch`
- `推論`: public API の出力は logical observable prediction である。理由は benchmark 側が `obs_flips` と直接比較しているため。
  - 根拠: `graphqec/decoder/pymatching.py`, 関数 `_process_batch`; `graphqec/benchmark/evaluate.py`, 関数 `benchmark_batch_acc`, `benchmark_sycamore_acc`

#### 3.1.3 SlidingWindowBPOSD

- `明示的記述`: パイプラインは `DEM -> detector_graph/obs_graph/priors -> temporal region decomposition -> per-window BPOSD -> accumulated total_error_hat -> obs_graph で logical prediction` である。
  - 根拠: `graphqec/decoder/_slidingwindow_bposd.py`, クラス `SlidingWindowBPOSD._initialize_regions_and_windows`, `SlidingWindowBPOSD.decode`
- `明示的記述`: syndrome を window ごとに更新しながら `total_error_hat` を commit していく。
  - 根拠: `graphqec/decoder/_slidingwindow_bposd.py`, 関数 `SlidingWindowBPOSD.decode`

### 3.2 Neural decoder pipeline

#### 3.2.1 GraphRNNDecoderV5A

- `明示的記述`: 入力 bit を embedding し、`cycle_encoder` で check-to-data message passing を行い、cycle ごとに recurrent decoder state を更新し、最後に `data_to_logical` 辺で readout する。
  - 根拠: `graphqec/decoder/nn/models.py`, クラス `GraphRNNDecoderV5._simple_forward`, `GraphRNNDecoderV5._incremental_forward`
- `明示的記述`: `encoding_syndromes`, `cycle_syndromes`, `readout_syndromes` を別々に扱う。
  - 根拠: `graphqec/decoder/nn/models.py`, クラス `QECCDecoder._decode`; クラス `GraphRNNDecoderV5._simple_forward`

#### 3.2.2 GraphLinearAttnDecoderV2A

- `明示的記述`: encoder 部分は GraphRNN 系と同様に syndrome を埋め込み、decoder 部で sequence 全体をまとめて処理し、最後に `data_to_logical` で readout する。
  - 根拠: `graphqec/decoder/nn/models.py`, クラス `GraphLinearAttnDecoderV2._simple_forward`, `GraphLinearAttnDecoderV2._incremental_forward`

### 3.3 surface-code variant の扱い方

- `明示的記述`: `SycamoreSurfaceCode._get_tanner_graph` は qubit 座標から data/check node を作り、近傍 4 方向で `data_to_check` を張る。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode._get_tanner_graph`
- `明示的記述`: initial / final time slice には `basis_mask` で絞った `masked_graph` を使い、中間 round には `default_graph` を使う。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode._get_tanner_graph`; `graphqec/qecc/code.py`, クラス `TemporalTannerGraph.__getitem__`
- `明示的記述`: `remove_irrelevent_detectors` は `since it is a CSS code` として detector を半分に落とす設計である。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `remove_irrelevent_detectors`
- `推論`: surface-code path では `X error -> Z-check matching`, `Z error -> X-check matching` を repo 独自に分離していない。classical decoder は full DEM を直接使い、neural decoder は single Tanner graph と detection-event sequence を使う。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `_get_decoder`; `graphqec/decoder/bposd.py`, クラス `BPOSD`; `graphqec/decoder/pymatching.py`, クラス `PyMatching`; `graphqec/decoder/nn/models.py`, クラス `QECCDecoder`
- `未確認`: XXZZ と XZZX を切り替える surface-code 専用分岐や明示的 naming は repo 内で未発見。

### 3.4 surface-code で使えない decoder

- `明示的記述`: `ConcatMatching` は `test_code.get_check_colors`, `get_check_basis`, `logical_basis` を要求する。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `_get_decoder`
- `明示的記述`: これらメソッド/属性は repo 内では `TriangleColorCode` にのみ定義され、`SycamoreSurfaceCode` には存在しない。
  - 根拠: `graphqec/qecc/color_code/sydney_color_code.py`, 関数 `get_check_colors`, `get_check_basis`; `graphqec/qecc/surface_code/google_block_memory.py`, クラス `SycamoreSurfaceCode`
- `推論`: `ConcatMatching` は surface-code path の有効 decoder ではない。
  - 根拠: 上記 2 点

## 4. 入出力インターフェースとデコードの運用形態

### 4.1 Classical decoder (`BPOSD`, `PyMatching`, `SlidingWindowBPOSD`)

- 入力データ
  - `明示的記述`: `raw_syndromes: np.ndarray` で shape は `[num_shots, num_detectors]`。
  - 根拠: `graphqec/decoder/bposd.py`, 関数 `BPOSD.decode`; `graphqec/decoder/pymatching.py`, 関数 `PyMatching.decode`; `graphqec/decoder/slidingwindow_bposd.py`, 関数 `SlidingWindowBPOSD.decode`
- syndrome の定義
  - `明示的記述`: stabilizer 値そのものではなく detector event である。benchmark は `sampler.sample(...)` の返す `syndromes` をそのまま decoder に渡し、実験データは `detection_events.b8` を読む。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `benchmark_batch_acc`, `benchmark_batch_time`, `benchmark_sycamore_acc`; `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `QECDataGoogleFormat.get_events`
  - `明示的記述`: final readout detector は final measurement record から `DETECTOR` 命令として再構成される。
  - 根拠: `graphqec/qecc/surface_code/google_utils.py`, 関数 `reverse_engineer_stim_circuit_then_split_into_two_parts`
- 出力データ
  - `明示的記述`: public API は logical observable prediction を返す。`BPOSD` と `SlidingWindowBPOSD` は内部 error estimate を `obs_graph` で observable flip に変換する。
  - 根拠: `graphqec/decoder/bposd.py`, 関数 `BPOSD.get_result`; `graphqec/decoder/_slidingwindow_bposd.py`, 関数 `SlidingWindowBPOSD.decode`
  - `推論`: `PyMatching` も logical observable prediction を返す。benchmark で `obs_flips` と直接比較しているため。
  - 根拠: `graphqec/decoder/pymatching.py`, 関数 `_process_batch`; `graphqec/benchmark/evaluate.py`, 関数 `benchmark_batch_acc`
- 運用形態
  - `推論`: full correction ではなく logical readout-only decoding / Pauli-frame inference に相当する。public interface から物理 correction を circuit に返す機構はない。
  - 根拠: `graphqec/decoder/bposd.py`, 関数 `BPOSD.get_result`; `graphqec/decoder/pymatching.py`, クラス `PyMatching`; `graphqec/benchmark/evaluate.py`, 関数 `benchmark_batch_acc`

### 4.2 Neural decoder

- 入力データ
  - `明示的記述`: public `decode` は `raw_syndromes: np.ndarray` を受け、内部で `encoding_syndromes`, `cycle_syndromes`, `readout_syndromes` に分割する。
  - 根拠: `graphqec/decoder/nn/models.py`, クラス `QECCDecoder._decode`
  - `明示的記述`: dataloader 経路では `(encoding_syndromes, cycle_syndromes, readout_syndromes)` の `torch.Tensor` を返す。
  - 根拠: `graphqec/decoder/nn/dataloader.py`, クラス `IncrementalSimDataset.__getitem__`, `ExperimentDataset.__getitem__`, `SimDataset.__getitem__`
- syndrome の定義
  - `明示的記述`: detector event を initial / cycle / readout に分解している。`readout_syndromes` は final detector 群であり、stabilizer の生値ではない。
  - 根拠: `graphqec/decoder/nn/dataloader.py`, クラス `ExperimentDataset.__getitem__`, `SimDataset.__getitem__`
- 出力データ
  - `明示的記述`: model 本体は logical flip logit を返し、public `decode` は sigmoid 後に bool 予測または `return_prob=True` で確率を返す。
  - 根拠: `graphqec/decoder/nn/models.py`, クラス `QECCDecoder._decode`
- 運用形態
  - `推論`: logical readout-only decoding である。物理 correction string や回路フィードバック API はない。
  - 根拠: `graphqec/decoder/nn/models.py`, クラス `QECCDecoder.decode`; `graphqec/benchmark/evaluate.py`, 関数 `benchmark_sycamore_acc`

### 4.3 surface code 専用の補足

- `明示的記述`: `TemporalTannerGraph` は `time_slice_graphs={0: masked_graph, -1: masked_graph}` を持ち、initial/final round に別 graph を与える。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode._get_tanner_graph`
- `推論`: これは初期 round と final readout round を、中間 syndrome round と同一視せず別扱いする設計である。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode._get_tanner_graph`; `graphqec/decoder/nn/models.py`, クラス `GraphRNNDecoderV5._simple_forward`

## 5. Neural network 系アルゴリズムの対応

### 5.1 inference 対応

- `明示的記述`: inference には対応している。`build_neural_decoder` が checkpoint を読み、`QECCDecoder.decode` が予測を返す。
  - 根拠: `graphqec/decoder/nn/train_utils.py`, 関数 `build_neural_decoder`; `graphqec/decoder/nn/models.py`, クラス `QECCDecoder.decode`

### 5.2 training 対応

- `明示的記述`: dataloader 構築関数と optimizer/scheduler utility は存在する。
  - 根拠: `graphqec/decoder/nn/train_utils.py`, 関数 `get_dataloaders`, `get_optimizer`, `construct_annealing_scheduler`
- `明示的記述`: `get_dataloaders` は `incremental`, `exp`, `sim` の 3 種の入力データタイプを想定している。
  - 根拠: `graphqec/decoder/nn/train_utils.py`, 関数 `get_dataloaders`
- `明示的記述`: ただし `exp` / `sim` 分岐は `from graphqec.decoder.nn.trainer import CurriculumTeacher` を import するが、この repo には `graphqec/decoder/nn/trainer.py` が存在しない。
  - 根拠: `graphqec/decoder/nn/train_utils.py`, 関数 `get_dataloaders`; repo ファイル一覧
- `推論`: end-to-end training support はこの checkout では不完全である。少なくとも `exp` / `sim` training path は追加コードなしでは実行不能の可能性が高い。
  - 根拠: 上記 2 点
- `未確認`: `incremental` path を使う外部 training script が別 repo や未同梱ファイルに存在するかは未確認。

### 5.3 合成訓練データの生成方法

- `incremental`
  - `明示的記述`: 0..`max_num_cycle` の incremental circuit 群に対して `stim.FlipSimulator` を個別実行し、各 cycle の logical flips と final readout detector 群をスタックする。
  - 根拠: `graphqec/decoder/nn/dataloader.py`, クラス `IncrementalSimDataset.__getitem__`
  - `推論`: surface-code path では Google-derived circuit noise からの synthetic data 生成であり、`physical_error_rate` による明示的 p-sweep ではない。
  - 根拠: `graphqec/decoder/nn/train_utils.py`, 関数 `get_dataloaders`; `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.get_syndrome_circuit`
- `sim`
  - `明示的記述`: 各 `num_cycle` ごとに単一 circuit を `FlipSimulator` でサンプルし、detector events と logical flips を返す。
  - 根拠: `graphqec/decoder/nn/dataloader.py`, クラス `SimDataset.__getitem__`
- `exp`
  - `明示的記述`: synthetic ではなく実験データを使う。各 cycle の `ExperimentDataset` を作り、先頭 `num_train` 件を train、残りを validation に分ける。
  - 根拠: `graphqec/decoder/nn/dataloader.py`, 関数 `get_exp_dataloaders`

## 6. ベンチマークの評価内容

### 6.1 repo に同梱される benchmark config / 結果 artifact の有無

- `明示的記述`: `configs/benchmark/` に同梱されている config は `ETHBBCode` と `TriangleColorCode` のみで、`SycamoreSurfaceCode` は含まれない。
  - 根拠: `configs/benchmark/*.json` の一覧; `rg -n '"code_type"' configs/benchmark`
- `明示的記述`: surface code 用の commit 済み benchmark 結果 artifact (`.csv`, `.pkl`, `.pt`) は repo 内で未発見。
  - 根拠: repo 全体のファイル一覧
- `推論`: surface code benchmark は「コードとしては存在する」が、「すぐ実行できる config と結果ファイル」は同梱されていない。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `benchmark_sycamore_acc`; `configs/benchmark/` の一覧

### 6.2 surface code で実装されている評価

#### 6.2.1 `sycamore_acc`

- `明示的記述`: 実験 data の per-cycle correctness を評価する。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `benchmark_sycamore_acc`
- `明示的記述`: post-processing は bootstrap により `accs`, `lfr`, `f0` を計算する。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `_process_single_sycamore_job_results`
- `推論`: これは logical memory 実験評価であり、1-shot readout benchmark ではない。理由は `test_cycles` に沿って複数 round の memory 成績を見て `fit_log_lfr` を当てているため。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `_process_single_sycamore_job_results`

#### 6.2.2 旧 benchmark (`evaluate_sycamore.py`)

- `明示的記述`: 複数 checkpoint の `return_prob=True` 出力を soft voting で平均し、その accuracy / `lfr` / `f0` を bootstrap で計算する。
  - 根拠: `graphqec/benchmark/evaluate_sycamore.py`, 関数 `benchmark_sycamore_acc`
- `推論`: 旧 benchmark は ensemble inference 用であり、現行 `evaluate.py` の `benchmark_sycamore_acc` より評価運用が広い。
  - 根拠: `graphqec/benchmark/evaluate_sycamore.py`, 関数 `init_experiment`, `benchmark_sycamore_acc`; `graphqec/benchmark/evaluate.py`, 関数 `benchmark_sycamore_acc`

### 6.3 generic benchmark が評価するもの

- `acc`
  - `明示的記述`: `Logical Error Rate`, `Rigid Logical Error Rate`, および per-round 版を出す。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `_process_single_acc_job_results`
  - `明示的記述`: 生データは `preds != obs_flips` で数え、logical-qubit 単位失敗と shot 単位失敗の両方を集計する。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `benchmark_batch_acc`
- `time`
  - `明示的記述`: decode latency (`Time mean`, `Time std`) を測る。正解率は評価しない。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `benchmark_batch_time`, `_process_single_time_job_results`

### 6.4 評価前提条件

- `sycamore_acc`
  - `明示的記述`: dataset 側に `parities` と `test_cycles` が必要。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `submit_benchmark`
  - `明示的記述`: `get_exp_data(r - 1, parity=...)` を呼ぶため、実データ側は odd cycle 系列に制限される。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `benchmark_sycamore_acc`; `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.get_exp_data`
  - `明示的記述`: `lfr` fit は `test_cycles[1:]` を使い、最初の cycle 点を fit から外す。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `_process_single_sycamore_job_results`; `graphqec/benchmark/evaluate_sycamore.py`, 関数 `benchmark_sycamore_acc`
  - `推論`: 評価対象は profile ごとに単一 basis (`Gd3X_*`, `Gd3Z_*`, `Gd5X`, `Gd5Z`) であり、logical Z 限定ではないが、1 回の task は 1 basis に固定される。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, クラス属性 `_PROFILES`
- generic `acc`
  - `明示的記述`: `num_fails_required` に達するまでサンプルを継続する停止条件である。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `benchmark_batch_acc`
- generic `time`
  - `明示的記述`: warm-up 後に `num_evaluation` と `batch_size` から決まる batch 数だけ測定する。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `benchmark_batch_time`

### 6.5 surface code benchmark に関する総括

- `明示的記述`: surface code について repo が直接サポートする評価は、主に Sycamore 実験 detection events に対する logical-memory decoding accuracy である。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `benchmark_sycamore_acc`
- `推論`: threshold 図や physical error rate sweep を伴う surface-code benchmark は、この checkout では主要経路ではない。generic benchmark code は存在するが、SycamoreSurfaceCode 側で `physical_error_rate` を消費しないためである。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `benchmark_batch_acc`, `benchmark_batch_time`; `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.get_dem`
