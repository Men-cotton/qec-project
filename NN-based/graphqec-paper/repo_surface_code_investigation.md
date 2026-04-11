# Repo Investigation: Syndrome Graph to Matching Output in `graphqec-paper`

更新方針:
- このファイルは「シンドロームグラフからマッチングを出力する過程」に絞る。
- 各主張では `明示的記述` と `推論` と `未確認` を分ける。
- 根拠として、該当ファイルパスと関数/クラス名を必ず添える。

## 1. 対象となるグラフ構造と実装範囲

### 1.1 結論

- `明示的記述`: この repo は surface-code 向けのマッチング処理を「実装またはラップ」している。README は `Sycamore Surface Codes` を supported code family に含め、classical decoder として `PyMatching` と `BPOSD` を列挙している。
  - 根拠: `README.md`, セクション `Scope`
- `明示的記述`: benchmark 側は surface-code 実装 `SycamoreSurfaceCode` に対して `PyMatching`, `BPOSD`, `SlidingWindowBPOSD` を生成する。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `_get_decoder`; `graphqec/qecc/surface_code/google_block_memory.py`, クラス `SycamoreSurfaceCode`
- `推論`: この repo で「マッチングを直接解く」経路は `PyMatching` ラッパーであり、`BPOSD` と `SlidingWindowBPOSD` は同じ detector graph 入力を使う別系統のグラフ復号器である。
  - 根拠: `graphqec/decoder/pymatching.py`, クラス `PyMatching`; `graphqec/decoder/bposd.py`, クラス `BPOSD`; `graphqec/decoder/slidingwindow_bposd.py`, クラス `SlidingWindowBPOSD`

### 1.2 surface code 実装の範囲

- `明示的記述`: repo 内で実際に読み込まれる surface-code 実装は `graphqec/qecc/surface_code/google_block_memory.py` の `SycamoreSurfaceCode` である。
  - 根拠: `graphqec/qecc/__init__.py`, `try: from .surface_code.google_block_memory import *`; `graphqec/qecc/surface_code/google_block_memory.py`, クラス `SycamoreSurfaceCode`
- `明示的記述`: `RotatedSurfaceCode` と `ZuchongzhiSurfaceCode` は条件付き import だが、対応ファイル `stim_block_memory.py` / `ustc_block_memory.py` はこの checkout に存在しない。
  - 根拠: `graphqec/qecc/__init__.py`, 条件付き import 節
- `明示的記述`: `SycamoreSurfaceCode` が持つ profile は `Gd3X_N/E/S/W`, `Gd3Z_N/E/S/W`, `Gd5X`, `Gd5Z` に限定される。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, クラス属性 `_PROFILES`
- `推論`: この checkout で確認できる surface-code マッチング対象は、Google/Sycamore 実験データに依存する CSS 系 surface code に限られる。generic な rotated planar XXZZ / XZZX を独自に生成するコードは同梱されていない。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, クラス `SycamoreSurfaceCode`; `graphqec/qecc/__init__.py`, 条件付き import 節
- `未確認`: `XXZZ` と `XZZX` の別を repo 内の明示記述だけで断定できる証拠は見つからない。

### 1.3 グラフの制約

- `明示的記述`: classical decoder の入力は `stim.DetectorErrorModel` であり、surface-code 側では `get_syndrome_circuit(num_cycle, parity=...).detector_error_model()` から得られる。
  - 根拠: `graphqec/qecc/code.py`, 抽象メソッド `QuantumCode.get_dem`; `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.get_dem`
- `明示的記述`: `SycamoreSurfaceCode.get_syndrome_circuit` は `num_cycle == 0` と `num_cycle > 0` を分け、`num_cycle > 0` では repeated syndrome rounds を含む full circuit を構築する。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.get_syndrome_circuit`
- `推論`: repo は code-capacity 専用の 2D matching graph を別実装していない。実際には `num_cycle` で長さの変わる detector-history graph を DEM として扱い、`num_cycle > 0` では 3D 時空間グラフ相当の入力を想定している。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.get_dem`, `SycamoreSurfaceCode.get_syndrome_circuit`; `graphqec/qecc/utils.py`, 関数 `dem_to_detector_graph`
- `明示的記述`: repo 側の detector-graph 表現は明示的な boundary node オブジェクトを持たず、`detector_graph[d, e]` 形式の detector-error incidence matrix を使う。
  - 根拠: `graphqec/qecc/utils.py`, 関数 `dem_to_detector_graph`
- `推論`: boundary に相当する「単一 detector を持つ error instruction」は incidence matrix の 1 列として保持できるが、repo 独自の named boundary node は作られない。boundary の意味付けは Stim DEM または外部ライブラリ側に委ねられている。
  - 根拠: `graphqec/qecc/utils.py`, 関数 `dem_to_detector_graph`; `graphqec/decoder/pymatching.py`, 関数 `_process_batch`
- `明示的記述`: 実験データ経路は odd cycle のみを許し、`num_cycle <= 24` に制限される。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.get_exp_data`, `SycamoreSurfaceCode.get_syndrome_circuit`

### 1.4 Capability Matrix

| Implementation | Graph dimension (2D/3D) | Boundary node support | Weighted edges support | Hyperedge support (for correlated/Y errors) | Dynamic graph generation | Parallel matching support |
| --- | --- | --- | --- | --- | --- | --- |
| `PyMatching` | `明示的記述`: `get_dem(num_cycle, ...)` の DEM を入力にする。`推論`: `num_cycle=0` 相当の短い graph と repeated-round の 3D detector history の両方を取り得る | `推論`: repo 側で boundary node を構築しない。DEM を `Matching.from_detector_error_model(dem)` にそのまま渡す | `推論`: repo 側は重みを計算せず DEM をそのまま渡すので、重み処理は外部ライブラリ依存 | `明示的記述`: repo 側で pair-edge 化せず DEM をそのまま渡す。`推論`: 多体 error の扱いは PyMatching 側依存 | `明示的記述`: `num_cycle` ごとに `get_dem` を呼んで decoder を作る | `明示的記述`: multiprocessing / submitit で batch 並列化 |
| `BPOSD` | `明示的記述`: DEM を `detector_graph` に変換して使う。`推論`: DEM が repeated rounds を含めば 3D detector-history graph をそのまま列化して扱う | `推論`: 明示 boundary node なし。単一 detector 列は保持可能 | `明示的記述`: `priors` を `channel_probs` として `BpOsdDecoder` に渡す | `明示的記述`: `dem_to_detector_graph` は detector/observable の任意列を保持する。`simplify_dem` も identical hyper-edge をまとめる | `明示的記述`: `get_dem(num_cycle, ...)` から毎回生成 | `明示的記述`: multiprocessing / submitit で batch 並列化 |
| `SlidingWindowBPOSD` | `明示的記述`: DEM を detector matrix に変換後、`num_detectors_per_cycle` で時系列 window に分割する。`推論`: repeated rounds を前提とした 3D detector-history 向け | `推論`: 明示 boundary node なし。`dem_to_detector_graph` の列表現に依存 | `明示的記述`: `priors` を window ごとに `BpOsdDecoder` へ渡す | `明示的記述`: 基底の `chk` 行列は任意列を保持する | `明示的記述`: `num_cycle` ごとに DEM を作り直す | `明示的記述`: multiprocessing / submitit で batch 並列化 |

Capability Matrix の根拠:
- `graphqec/benchmark/evaluate.py`, 関数 `_get_decoder`
- `graphqec/decoder/pymatching.py`, クラス `PyMatching`
- `graphqec/decoder/bposd.py`, クラス `BPOSD`
- `graphqec/decoder/slidingwindow_bposd.py`, クラス `SlidingWindowBPOSD`
- `graphqec/decoder/_slidingwindow_bposd.py`, クラス `SlidingWindowBPOSD`
- `graphqec/qecc/utils.py`, 関数 `dem_to_detector_graph`, `simplify_dem`

## 2. グラフ構築とエッジ重みの計算

### 2.1 シンドロームグラフはどう構築されるか

- `明示的記述`: surface-code 実装は外部 Sycamore データセットの `circuit_detector_error_model.dem` と `circuit_noisy.stim` / `circuit_ideal.stim` を読み込む。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, クラス `QECDataGoogleFormat.__init__`, `QECDataGoogleFormat.get_dem`, `QECDataGoogleFormat.get_circuit`
- `明示的記述`: `SycamoreSurfaceCode.get_dem` は `get_syndrome_circuit(...).detector_error_model()` を返す。surface code 用に detector graph を handcraft する関数はない。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.get_dem`
- `明示的記述`: `dem_to_detector_graph` は DEM を `detector_graph: bool[num_detectors, num_errors]`, `obs_graph: bool[num_observables, num_errors]`, `priors: float[num_errors]` に変換する。
  - 根拠: `graphqec/qecc/utils.py`, 関数 `dem_to_detector_graph`
- `推論`: この repo における「シンドロームグラフ」は NetworkX のような明示 adjacency graph ではなく、Stim DEM を列ごとの error event incidence matrix に落とした表現である。
  - 根拠: `graphqec/qecc/utils.py`, 関数 `dem_to_detector_graph`

### 2.2 Code capacity (2D) と repeated rounds (3D) の差

- `明示的記述`: `SycamoreSurfaceCode.get_syndrome_circuit(num_cycle=0)` は `_cycle0` を返し、`num_cycle>0` では `construct_full_circuit_from_blocks(..., num_cycle)` を使う。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.get_syndrome_circuit`
- `明示的記述`: dataloader は raw detector events を `encoding_syndromes`, `cycle_syndromes`, `readout_syndromes` に分割し、`cycle_syndromes` を `[batch, num_cycle, num_detectors_per_round]` に reshape する。
  - 根拠: `graphqec/decoder/nn/dataloader.py`, クラス `ExperimentDataset.__getitem__`, `SimDataset.__getitem__`, `IncrementalSimDataset.__getitem__`
- `推論`: repeated rounds の有無は「別 graph generator」ではなく、raw detector vector の長さと reshape 規則、および DEM の detector 数の違いとして表現される。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `SycamoreSurfaceCode.get_syndrome_circuit`; `graphqec/decoder/nn/dataloader.py`, 各 `__getitem__`
- `未確認`: phenomenological noise と circuit-level noise を surface-code 専用 API で切り替えるフラグは見当たらない。repo 内では Google 実データ由来 circuit/DEM を使う経路が主である。

### 2.3 エッジ重みの計算・保持

- `明示的記述`: `dem_to_detector_graph` は各 `error(p)` instruction の確率 `p` をそのまま `priors[i]` に格納する。
  - 根拠: `graphqec/qecc/utils.py`, 関数 `dem_to_detector_graph`
- `明示的記述`: `BPOSD` と `SlidingWindowBPOSD` はその `priors` を `BpOsdDecoder(..., channel_probs=priors)` に渡す。
  - 根拠: `graphqec/decoder/bposd.py`, 関数 `_create_decoder`; `graphqec/decoder/_slidingwindow_bposd.py`, 関数 `SlidingWindowBPOSD.decode`
- `明示的記述`: `simplify_dem` は同一 hyper-edge の確率を `1 - (1 - old_p) * (1 - current_p)` で合成する。
  - 根拠: `graphqec/qecc/utils.py`, 関数 `simplify_dem`
- `推論`: repo 内では対数重み化や整数化への丸めは行っていない。保持されるのは float の error probability であり、PyMatching 側の重み化も repo 内コードでは明示されない。
  - 根拠: `graphqec/qecc/utils.py`, 関数 `dem_to_detector_graph`; `graphqec/decoder/pymatching.py`, 関数 `_process_batch`

### 2.4 相関誤りと hyperedge

- `明示的記述`: `dem_to_detector_graph` は 1 つの error instruction が何個の detector / observable を含んでも、その列をそのまま保持する。
  - 根拠: `graphqec/qecc/utils.py`, 関数 `dem_to_detector_graph`
- `明示的記述`: `simplify_dem` の docstring は「same hyper-edge」をまとめる処理であると明記している。
  - 根拠: `graphqec/qecc/utils.py`, 関数 `simplify_dem`
- `推論`: BPOSD / SlidingWindowBPOSD の repo 内表現は「独立した X graph と Z graph への完全分離」ではなく、DEM の列単位 error event をそのまま扱う設計である。
  - 根拠: `graphqec/qecc/utils.py`, 関数 `dem_to_detector_graph`; `graphqec/decoder/bposd.py`, クラス `BPOSD`; `graphqec/decoder/_slidingwindow_bposd.py`, クラス `SlidingWindowBPOSD`
- `明示的記述`: `PyMatching` wrapper も repo 側で X/Z 分離や hyperedge 分解をせず、`Matching.from_detector_error_model(dem)` に丸ごと委譲する。
  - 根拠: `graphqec/decoder/pymatching.py`, 関数 `_process_batch`
- `未確認`: Y error や multi-detector correlated event を PyMatching が内部でどう扱うかは、この repo 内コードだけでは確認できない。

## 3. マッチングアルゴリズムの概要

### 3.1 PyMatching

- `明示的記述`: `PyMatching` wrapper のコアは `Matching.from_detector_error_model(dem)` と `decoder.decode_batch(batch_syndromes)` である。
  - 根拠: `graphqec/decoder/pymatching.py`, 関数 `_process_batch`
- `明示的記述`: repo は PyMatching を外部依存として使う。
  - 根拠: `README.md`, セクション `Scope`; `pyproject.toml`, dependencies; `graphqec/decoder/pymatching.py`, `from pymatching import Matching`
- `推論`: repo 内に MWPM 本体のスクラッチ実装はない。matching の中核アルゴリズムは PyMatching 依存である。
  - 根拠: `graphqec/decoder/pymatching.py`, クラス `PyMatching`

### 3.2 BPOSD

- `明示的記述`: `BPOSD` は `ldpc.BpOsdDecoder` を使い、`bp_method="minimum_sum"` と `osd_method="osd_cs"` を指定する。
  - 根拠: `graphqec/decoder/bposd.py`, 関数 `_create_decoder`
- `明示的記述`: pipeline は `DEM -> simplify_dem -> dem_to_detector_graph -> BpOsdDecoder.decode(syndrome) -> errs @ obs_graph.T mod 2` である。
  - 根拠: `graphqec/decoder/bposd.py`, クラス `BPOSD.__init__`, `BPOSD.get_result`
- `推論`: これは MWPM ではなく、belief propagation + ordered statistics decoding による detector-graph 復号である。
  - 根拠: `graphqec/decoder/bposd.py`, 関数 `_create_decoder`

### 3.3 SlidingWindowBPOSD

- `明示的記述`: `SlidingWindowBPOSD` は `dem_to_detector_graph` で得た `chk` 行列を `num_detectors_per_cycle` と `half_cycle` に基づいて時間窓へ分割する。
  - 根拠: `graphqec/decoder/_slidingwindow_bposd.py`, クラス `SlidingWindowBPOSD.__init__`, `_initialize_regions_and_windows`
- `明示的記述`: 各窓で `BpOsdDecoder` を生成し、部分 syndrome を decode し、commit 済み部分だけ `total_error_hat` に反映する。
  - 根拠: `graphqec/decoder/_slidingwindow_bposd.py`, 関数 `SlidingWindowBPOSD.decode`
- `推論`: repeated syndrome rounds を持つ長い detector history を低メモリに処理するための近似的な時系列分割復号器であり、pair matching を直接返す実装ではない。
  - 根拠: `graphqec/decoder/_slidingwindow_bposd.py`, クラス `SlidingWindowBPOSD`

## 4. 入出力インターフェースとデータ構造

### 4.1 入力データ: グラフ

- `明示的記述`: classical decoder の graph 入力型は `stim.DetectorErrorModel` である。
  - 根拠: `graphqec/decoder/bposd.py`, クラス `BPOSD.__init__`; `graphqec/decoder/pymatching.py`, クラス `PyMatching.__init__`; `graphqec/decoder/slidingwindow_bposd.py`, クラス `SlidingWindowBPOSD.__init__`
- `明示的記述`: repo 内部ではそれを `detector_graph`, `obs_graph`, `priors` の NumPy 配列へ変換する。
  - 根拠: `graphqec/qecc/utils.py`, 関数 `dem_to_detector_graph`
- `明示的記述`: neural decoder 用 graph 入力は `TemporalTannerGraph` であり、`default_graph: TannerGraph` と `time_slice_graphs: Dict[int, TannerGraph]` を持つ。
  - 根拠: `graphqec/qecc/code.py`, dataclass `TemporalTannerGraph`
- `明示的記述`: `TannerGraph` は `data_nodes`, `check_nodes`, `data_to_check`, `data_to_logical` を `np.ndarray | torch.Tensor` で持つ。
  - 根拠: `graphqec/qecc/code.py`, dataclass `TannerGraph`

### 4.2 入力データ: シンドローム

- `明示的記述`: classical decoder の public `decode` は `raw_syndromes: np.ndarray` を受け、shape は `[num_shots, num_detectors]` である。
  - 根拠: `graphqec/decoder/bposd.py`, 関数 `BPOSD.decode`; `graphqec/decoder/pymatching.py`, 関数 `PyMatching.decode`; `graphqec/decoder/slidingwindow_bposd.py`, 関数 `SlidingWindowBPOSD.decode`
- `明示的記述`: surface-code 実験データは `detection_events.b8` から `stim.read_shot_data_file(..., num_detectors=self.raw_dem.num_detectors)` で読む。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `QECDataGoogleFormat.get_events`
- `推論`: decoder に渡されるのは stabilizer 値そのものではなく detector event vector である。
  - 根拠: `graphqec/qecc/surface_code/google_block_memory.py`, 関数 `QECDataGoogleFormat.get_events`; `graphqec/benchmark/evaluate.py`, 関数 `benchmark_batch_acc`, `benchmark_sycamore_acc`

### 4.3 出力データ

- `明示的記述`: `BPOSD.get_result` は `errs @ obs_graph.T mod 2` で observable flip 予測を返す。
  - 根拠: `graphqec/decoder/bposd.py`, 関数 `BPOSD.get_result`
- `明示的記述`: `_slidingwindow_bposd.SlidingWindowBPOSD.decode` も最後に `(total_error_hat @ self.obs.T) % 2` を返す。
  - 根拠: `graphqec/decoder/_slidingwindow_bposd.py`, 関数 `SlidingWindowBPOSD.decode`
- `明示的記述`: `PyMatching.get_result` は `decode_batch` の返り値を連結して `bool` 化して返す。
  - 根拠: `graphqec/decoder/pymatching.py`, 関数 `PyMatching.get_result`
- `推論`: repo の public interface は「マッチングされた detector node のペア」や「採用 edge のリスト」を返さず、logical observable prediction を返す。
  - 根拠: `graphqec/decoder/bposd.py`, 関数 `BPOSD.get_result`; `graphqec/decoder/pymatching.py`, 関数 `PyMatching.get_result`; `graphqec/decoder/_slidingwindow_bposd.py`, 関数 `SlidingWindowBPOSD.decode`
- `未確認`: PyMatching 内部で得られる matching pair / correction edge を repo wrapper が取得する API は見当たらない。

### 4.4 コア関数・クラスのシグネチャ

- `graphqec/qecc/code.py`
  - `QuantumCode.get_dem(self, num_cycle, **noise_kwargs) -> stim.DetectorErrorModel`
  - `QuantumCode.get_tanner_graph(self) -> TemporalTannerGraph`
- `graphqec/qecc/utils.py`
  - `dem_to_detector_graph(dem: stim.DetectorErrorModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
- `graphqec/decoder/bposd.py`
  - `BPOSD.decode(self, raw_syndromes: np.ndarray, *, batch_size=100, non_blocking=False) -> np.ndarray[np.bool_]`
  - `BPOSD.get_result(self)`
- `graphqec/decoder/pymatching.py`
  - `PyMatching.decode(self, raw_syndromes: np.ndarray, *, batch_size=100, non_blocking=False) -> np.ndarray[np.bool_]`
  - `PyMatching.get_result(self)`
- `graphqec/decoder/_slidingwindow_bposd.py`
  - `SlidingWindowBPOSD.decode(self, syndromes: np.ndarray) -> np.ndarray`
- `graphqec/decoder/slidingwindow_bposd.py`
  - `SlidingWindowBPOSD.decode(self, raw_syndromes: np.ndarray, *, batch_size=100, non_blocking=False) -> np.ndarray[np.bool_]`

## 5. Neural network 系アルゴリズムの適用

### 5.1 何に NN が使われているか

- `明示的記述`: README が neural decoders として `GraphRNNDecoderV5A` と `GraphLinearAttnDecoderV2A` を挙げる。
  - 根拠: `README.md`, セクション `Scope`
- `明示的記述`: benchmark 側は classical decoder 名以外を neural decoder とみなし、`test_code.get_tanner_graph().to(device)` を `build_neural_decoder` に渡す。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `_get_decoder`
- `推論`: NN は matching edge や matching pair を直接出力していない。出力は logical flip logit / bool 予測であり、edge-weight inference API もない。
  - 根拠: `graphqec/decoder/nn/models.py`, クラス `QECCDecoder._decode`, `GraphRNNDecoderV5._simple_forward`, `GraphLinearAttnDecoderV2._simple_forward`

### 5.2 グラフとシンドロームをどうテンソル化するか

- `明示的記述`: `TemporalTannerGraph` は `to(device)` で NumPy / Torch 間変換でき、decoder は `tanner_graph[0]`, `tanner_graph[...]`, `tanner_graph[-1]` を使い分ける。
  - 根拠: `graphqec/qecc/code.py`, dataclass `TemporalTannerGraph`; `graphqec/decoder/nn/models.py`, クラス `QECCDecoder`, `GraphRNNDecoderV5._simple_forward`
- `明示的記述`: raw detector vector は `encoding_syndromes`, `cycle_syndromes`, `readout_syndromes` に分割される。
  - 根拠: `graphqec/decoder/nn/models.py`, 関数 `QECCDecoder._decode`; `graphqec/decoder/nn/dataloader.py`, 各 `__getitem__`
- `明示的記述`: `GraphRNNDecoderV5` と `GraphLinearAttnDecoderV2` は syndrome bit を embedding し、`data_to_check` と `data_to_logical` を使って message passing / readout を行う。
  - 根拠: `graphqec/decoder/nn/models.py`, クラス `GraphRNNDecoderV5._simple_forward`, `GraphLinearAttnDecoderV2._simple_forward`
- `推論`: NN 系は「matching graph を解く」のではなく、「固定 Tanner graph 上で syndrome sequence から logical observable を直接分類する」設計である。
  - 根拠: `graphqec/decoder/nn/models.py`, クラス `QECCDecoder`

## 6. マッチング処理のパフォーマンス・ベンチマーク

### 6.1 何が評価されているか

- `明示的記述`: generic `time` benchmark は decode latency の mean/std を測る。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `benchmark_batch_time`, `_process_single_time_job_results`
- `明示的記述`: generic `acc` benchmark は `preds != obs_flips` に基づく logical error rate を評価する。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `benchmark_batch_acc`, `_process_single_acc_job_results`
- `明示的記述`: `benchmark_sycamore_acc` は experiment 由来 detection events に対する logical-memory decoding accuracy を測る。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `benchmark_sycamore_acc`

### 6.2 matching solver としての評価か

- `推論`: repo にある評価の主対象は「logical observable prediction の精度 / 時間」であり、「matching pair の最適コスト」や「厳密解との差」ではない。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `benchmark_batch_acc`, `benchmark_batch_time`, `benchmark_sycamore_acc`
- `明示的記述`: `time` benchmark も decoder API の実行時間を測るだけで、グラフサイズや syndrome density ごとの詳細 scaling 指標は直接出力しない。
  - 根拠: `graphqec/benchmark/evaluate.py`, 関数 `benchmark_batch_time`, `_process_single_time_job_results`
- `未確認`: matching edge 数、matching cost、メモリ使用量、厳密 MWPM との差を surface-code matching solver として評価する専用スクリプトは repo 内で未発見。

## 7. 未対応理由の切り分け

- generic rotated-planar / toric surface-code matching
  - 判定: `この checkout では未確認`
  - 根拠: `graphqec/qecc/__init__.py` は条件付き import を置くが、対応 surface-code 実装ファイルが欠けている
- explicit boundary-node graph builder
  - 判定: `設計上対象外の可能性が高い`
  - 根拠: repo は Stim DEM を中心に扱い、`dem_to_detector_graph` でも boundary node を独立データ構造にしていない
- matching pair / edge list の出力
  - 判定: `未実装の可能性が高い`
  - 根拠: public decoder API は logical observable prediction しか返さない
- neural network による matching edge 直接予測
  - 判定: `未実装`
  - 根拠: `graphqec/decoder/nn/models.py` は logical flip classification を返し、edge-level 出力 head がない
