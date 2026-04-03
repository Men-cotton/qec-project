# PyMatching repo 調査メモ

最終更新段階: 6 / 6

## 調査方針

- ルール: 各主張について、可能な限り `file path` と `function/class` を併記する。
- ルール: `明示的記述` と `推論` を分離する。
- ルール: 該当証拠が見つからない場合は `未確認` と記す。
- ルール: 未対応理由は `未実装` と `前提アーキテクチャ上の対象外` を区別する。

---

## 1. Surface code の対応状況と実装範囲

### 1.1 結論

- 結論: この repo は surface-code 系 QEC のデコードに `対応している`。ただし、surface code 専用実装を中心にした repo ではなく、`汎用 MWPM デコーダ` が `Stim の DetectorErrorModel` または `check matrix` を読み込んで surface-code 由来の問題をデコードする構成である。
  - 根拠:
    - `README.md` / `Matching.from_detector_error_model`, `Matching.from_stim_circuit` を使う usage 例で `stim.Circuit.generated("surface_code:rotated_memory_x", ...)` を直接使用。
    - `src/pymatching/matching.py` / `class Matching`, `Matching.from_detector_error_model`, `Matching.from_stim_circuit`, `Matching.load_from_check_matrix`
    - `tests/matching/load_from_stim_test.py` / `test_load_from_stim_objects`
    - `tests/matching/decode_test.py` / `test_surface_code_solution_weights`, `test_surface_code_solution_weights_with_correlations`
    - `benchmarks/surface_codes/README.md` / `save_benchmark_circuit`, `time_surface_code_circuit`
    - `docs/toric-code-example.ipynb` / `toric_code_x_stabilisers`, `toric_code_x_logicals`

### 1.2 対象ディレクトリ

- `src/pymatching/`
  - 役割: 汎用 MWPM デコーダ本体。surface code 固有ではなく、graph / check matrix / DEM を受け取って復号する。
  - 根拠:
    - `src/pymatching/matching.py` / `class Matching`
    - `src/pymatching/sparse_blossom/driver/user_graph.cc` / `detector_error_model_to_user_graph`
    - `src/pymatching/sparse_blossom/driver/mwpm_decoding.cc` / `decode_detection_events`, `decode_detection_events_to_edges`
- `benchmarks/surface_codes/`
  - 役割: surface code ベンチマーク用 Stim circuit と説明。
  - 根拠:
    - `benchmarks/surface_codes/README.md` / `save_benchmark_circuit`, `time_surface_code_circuit`
- `tests/matching/`, `src/pymatching/sparse_blossom/driver/`
  - 役割: surface-code / toric-code 入力を使った回帰テスト。
  - 根拠:
    - `tests/matching/load_from_stim_test.py` / `test_load_from_stim_objects`
    - `tests/matching/decode_test.py` / `test_surface_code_solution_weights`, `test_surface_code_solution_weights_with_correlations`
    - `src/pymatching/sparse_blossom/driver/mwpm_decoding.test.cc` / `InvalidSyndromeForToricCode`
- `docs/`
  - 役割: toric code と Stim surface code を使う usage notebook。
  - 根拠:
    - `docs/toric-code-example.ipynb` / `toric_code_x_stabilisers`, `toric_code_x_logicals`
    - `docs/getting-started.ipynb` / `surface_code_circuit`, `get_logical_error_rate_pymatching`

### 1.3 repo 内で明示される code family

- `rotated surface code memory experiment in X basis`
  - 明示的記述:
    - `stim.Circuit.generated("surface_code:rotated_memory_x", ...)` が README, docs, tests, benchmarks で繰り返し使われる。
  - 根拠:
    - `README.md` / surface-code usage example
    - `src/pymatching/matching.py` / `Matching.from_detector_error_model`, `Matching.from_stim_circuit` の docstring 例
    - `tests/matching/load_from_stim_test.py` / `test_load_from_stim_objects`
    - `benchmarks/surface_codes/README.md` / `save_benchmark_circuit`, `time_surface_code_circuit`
    - `docs/getting-started.ipynb` / `surface_code_circuit`
- `toric code (unrotated / periodic surface-code family)`
  - 明示的記述:
    - notebook が toric code を parity-check matrix から構成する例を持つ。
    - test data に `toric_code_unrotated_memory_x_5_0.005.dem` がある。
  - 根拠:
    - `docs/toric-code-example.ipynb` / `toric_code_x_stabilisers`, `toric_code_x_logicals`
    - `src/pymatching/sparse_blossom/driver/mwpm_decoding.test.cc` / `InvalidSyndromeForToricCode`
    - `data/toric_code_unrotated_memory_x_5_0.005.dem` / data fixture

- `XZZX`, `XXZZ`, lattice-surgery-specific code family`
  - 結論: `未確認`
  - 証拠の不在:
    - repo 内主要エントリポイントと surface-code 関連ディレクトリを検索した範囲で `XZZX` / `XXZZ` / `lattice surgery` を実装名として使う箇所は見つからない。
  - 根拠:
    - `README.md`
    - `src/pymatching/matching.py` / `class Matching`
    - `benchmarks/surface_codes/README.md`
    - `tests/matching/load_from_stim_test.py` / `test_load_from_stim_objects`
    - `docs/toric-code-example.ipynb` / `toric_code_x_stabilisers`
  - 判定理由:
    - `未実装` と断定はしない。repo は汎用 graph/DEM decoder なので、外部入力として与えれば理論上読み込める可能性はあるが、repo 内に surface-code variant としての明示例・生成器・テストがないため `未確認` とする。

### 1.4 実装の制約

- `surface-code 固有ロジックは薄く、入力依存`
  - 明示的記述:
    - `Matching` は NetworkX / rustworkx / check matrix / Stim DEM / Stim circuit を読み込む汎用 API である。
  - 根拠:
    - `src/pymatching/matching.py` / `class Matching.__init__`, `Matching.load_from_check_matrix`, `Matching.from_detector_error_model`, `Matching.from_stim_circuit`
  - 含意:
    - surface-code の「形」「境界」「logical qubit 数」の多くは repo 固有ロジックではなく、入力された graph/DEM 側で決まる。

- `single logical qubit に固定された surface-code 実装` ではない
  - 明示的記述:
    - decoder の出力次元は `num_fault_ids` / `num_observables` に依存する。
    - check-matrix API は `faults_matrix` を受け取り、複数 observables を直接予測できる。
  - 根拠:
    - `src/pymatching/matching.py` / `Matching.decode_batch`, `Matching.load_from_check_matrix`
    - `src/pymatching/sparse_blossom/driver/user_graph.h` / `UserGraph(size_t num_nodes, size_t num_observables)`, `get_num_observables`
    - `docs/toric-code-example.ipynb` / `toric_code_x_logicals`, `num_decoding_failures`, `num_decoding_failures_vectorised`
  - ただし:
    - repo に同梱された `surface_code:rotated_memory_x` 例は single-observable memory task に偏っている。
  - 根拠:
    - `tests/matching/decode_test.py` / `test_surface_code_solution_weights` で `expected_observables_arr` の shape を `(shots.shape[0], 1)` として扱う。

- `odd distance 限定` ではない
  - 明示的記述:
    - benchmark データに `distance=50` が含まれる。
    - `Matching.from_stim_circuit` / `from_detector_error_model` / `load_from_check_matrix` に odd-distance 制約チェックは見当たらない。
  - 根拠:
    - `benchmarks/surface_codes/surface_code_rotated_memory_x_p_0.001_d_5_7_9_13_17_23_29_39_50_both_bases/`
    - `src/pymatching/matching.py` / `Matching.from_stim_circuit`, `Matching.from_detector_error_model`, `Matching.load_from_check_matrix`

- `open boundary / planar patch / rectangular patch`
  - 明示的記述:
    - repo 自身は geometry を直接表現せず、boundary edge を generic に扱う。
  - 根拠:
    - `src/pymatching/matching.py` / `Matching.add_boundary_edge`, `Matching.set_boundary_nodes`, `Matching.load_from_check_matrix`
    - `src/pymatching/sparse_blossom/driver/user_graph.cc` / `add_or_merge_boundary_edge`, `set_boundary`
  - 推論:
    - open boundary や rectangular patch を repo のコアが禁止している証拠はない。
  - 未確認:
    - repo 内に「rectangular rotated planar patch」を生成・検証する surface-code 固有コードは見つからない。
    - `surface_code:rotated_memory_x` の境界形状そのものは repo 内で定義されておらず、Stim 側タスク名に依存している。

### 1.5 Capability Matrix

| code family | patch shape | single/multi logical qubit | boundaries | odd distance restriction | repeated syndrome rounds | measurement error support | active correction support | lattice surgery | benchmark scripts | neural decoder |
|---|---|---|---|---|---|---|---|---|---|---|
| Stim `surface_code:rotated_memory_x` | 明示: square distance-`d` examplesのみ。`distance` を単独 scalar で指定。rectangular は未確認 | 明示: 同梱 examples/tests は single logical observable。汎用 API として multi-observable は可能 | 明示: repo は DEM/graph の boundary edge を扱える。rotated patch の具体境界形状は repo 内では未確認 | 明示: repo 制約なし。benchmark に `d=50` あり | 明示: あり。`rounds=distance` を使用 | 明示: あり。`before_measure_flip_probability` と DEM の boundary/time-like detector events | 明示: logical observable 予測が主。回路への能動フィードバック実装は未確認 | 未確認 | 明示: あり。`benchmarks/surface_codes/README.md` | 明示: なし |
| Toric code check-matrix example | 明示: torus lattice size `L` | 明示: multi-observable を扱う `faults_matrix` 例あり | 明示: toric code 名称上 periodic。notebook は boundary なし構成 | 未確認。example は odd `L` を使うが API 制約の証拠なし | 明示: perfect-syndrome 版あり、phenomenological repetition 版あり | 明示: phenomenological 版あり | 明示: physical-frame prediction も logical prediction も可能。active feedback は未確認 | 未確認 | notebook はあるが benchmark script は未確認 | 明示: なし |

### 1.6 非対応・未確認項目の理由

- `XZZX / XXZZ`
  - 判定: `未確認`
  - 理由分類: `未実装と断定はしない`
  - 理由:
    - repo の設計は「任意の graphlike Tanner graph / DEM を decode する」汎用 decoder なので、variant 名を repo 内に持たないまま外部入力として扱える余地がある。
    - ただし repo 内に variant 名・生成器・テスト・ベンチマークが見当たらないため、repo 付属機能としては確認不能。

- `lattice surgery`
  - 判定: `未確認`
  - 理由分類: `surface-code アプリケーション層が repo の主対象外`
  - 理由:
    - repo のコアは graph/DEM 復号器であり、lattice surgery の回路生成・操作手順・専用評価スクリプトは見つからない。
    - `benchmarks/surface_codes/README.md` に twist-defect を背景説明として触れる文はあるが、実装ではない。

---

## 2. 対象ノイズモデル

### 2.1 デコーダ実装が仮定するノイズモデル

#### 2.1.1 標準 MWPM の前提

- 結論: 標準 MWPM 実装の基本前提は `independent + graphlike` である。
  - 明示的記述:
    - README が「error mechanisms are independent, as well as graphlike」と明記。
    - check-matrix API は各 column が 1 つまたは 2 つの 1 を持つ graphlike error を要求。
    - DEM API は 1 または 2 detection events を起こす graphlike error を edge として読む。
  - 根拠:
    - `README.md`
    - `src/pymatching/matching.py` / `class Matching`, `Matching.load_from_check_matrix`, `Matching.from_detector_error_model`
    - `src/pymatching/sparse_blossom/driver/user_graph.h` / `merge_weights`

- 結論: 標準 MWPM の weight は独立エラーを仮定した `log((1-p)/p)` 型で扱われる。
  - 明示的記述:
    - README が independent な error probability `p_j` に対し `log((1-p_j)/p_j)` を推奨。
    - C++ 側の edge merge も independent な parallel error を前提に `merge_weights` を定義。
  - 根拠:
    - `README.md`
    - `src/pymatching/sparse_blossom/driver/user_graph.h` / `merge_weights`
    - `src/pymatching/sparse_blossom/driver/user_graph.cc` / `handle_dem_instruction`, `merge_edge_or_boundary_edge`

#### 2.1.2 measurement error と phenomenological model

- 結論: check-matrix 経路では `phenomenological noise` に対応する。
  - 明示的記述:
    - `Matching.decode` docstring に「qubits and measurements both suffering bit-flip errors」とある。
    - `repetitions`, `timelike_weights`, `measurement_error_probabilities` を用いて time dimension を追加する。
  - 根拠:
    - `src/pymatching/matching.py` / `Matching.decode`, `Matching.load_from_check_matrix`
    - `docs/toric-code-example.ipynb` / `num_decoding_failures_noisy_syndromes`

- measurement error の扱い
  - 明示的記述:
    - 入力 syndrome は stabilizer 値そのものではなく `difference syndrome` を要求する。
    - 最終 round は perfect measurement であるべきだと notebook が明記。
  - 根拠:
    - `src/pymatching/matching.py` / `Matching.decode`
    - `docs/toric-code-example.ipynb` / `num_decoding_failures_noisy_syndromes`

- データ誤りのみのモデル
  - 明示的記述:
    - perfect-syndrome の check-matrix 例は qubit error のみをサンプリングして `syndrome = H@noise % 2` を計算。
  - 根拠:
    - `README.md`
    - `docs/toric-code-example.ipynb` / `num_decoding_failures_via_physical_frame_changes`, `num_decoding_failures`

#### 2.1.3 circuit-level DEM の前提

- 結論: Stim DEM 経路は `circuit-level noise` を受け取れるが、decoder 本体が直接扱うのは `graphlike` error か、その分解結果である。
  - 明示的記述:
    - `Matching.from_detector_error_model` docstring は DEM を circuit-level noise model の表現と説明。
    - graphlike でない error instruction は、decomposition がなければ ignore される。
  - 根拠:
    - `src/pymatching/matching.py` / `Matching.from_detector_error_model`
    - `README.md`

- measurement error の有無
  - 結論: DEM 経路では、measurement error は detector event を含む edge/boundary edge として取り込まれる。repo は「measurement error 専用分岐」を持つのではなく、DEM に埋め込まれた detector connectivity を generic に decode する。
  - 根拠:
    - `src/pymatching/matching.py` / `Matching.from_detector_error_model`
    - `src/pymatching/sparse_blossom/driver/user_graph.cc` / `handle_dem_instruction`

#### 2.1.4 相関誤り・Y 誤り・biased noise

- 結論: 標準 MWPM は `独立 edge` 前提であり、相関を直接は使わない。
  - 根拠:
    - `README.md`
    - `src/pymatching/matching.py` / `Matching.from_detector_error_model`, `Matching.decode`

- 結論: `enable_correlations=True` の two-pass correlated matching は、`decompose_errors=True` で分解された hyperedge 由来の `edge correlation` を扱う。
  - 明示的記述:
    - README は Y error in the surface code を例に挙げる。
    - `iter_dem_instructions_include_correlations` は分解された複数 component から joint probabilities を作り implied weight を生成する。
  - 根拠:
    - `README.md`
    - `src/pymatching/matching.py` / `Matching.decode`, `Matching.decode_batch`
    - `src/pymatching/sparse_blossom/driver/user_graph.h` / `iter_dem_instructions_include_correlations`
    - `src/pymatching/sparse_blossom/driver/user_graph.cc` / `populate_implied_edge_weights`
    - `src/pymatching/sparse_blossom/driver/mwpm_decoding.cc` / `decode_detection_events`, `decode_detection_events_to_edges_with_edge_correlations`

- Y 誤りの扱い
  - 明示的記述:
    - README と docstring は `Y error in the surface code` を correlated matching の典型例として挙げる。
    - test は `PAULI_CHANNEL_1(0, p, 0)` を使った回路から correlated matching を構成する。
  - 根拠:
    - `README.md`
    - `src/pymatching/matching.py` / `Matching.decode`, `Matching.decode_batch`
    - `tests/matching/decode_test.py` / `test_correlated_matching_handles_single_detector_components`

- 未サポートの相関
  - 結論: `undecomposed hyperedge` は correlated path で未対応。
  - 明示的記述:
    - `iter_dem_instructions_include_correlations` は 3 detector 以上の undecomposed component で例外を投げる。
    - test も `decompose_errors=False` で correlations enabled のロード失敗を確認する。
  - 根拠:
    - `src/pymatching/sparse_blossom/driver/user_graph.h` / `iter_dem_instructions_include_correlations`
    - `tests/matching/decode_test.py` / `test_use_correlations_without_decompose_errors_raises_value_error`

- 確率 > 0.5 の扱い
  - 結論: correlations enabled path では `p > 0.5` を不許可。
  - 根拠:
    - `src/pymatching/sparse_blossom/driver/user_graph.h` / `iter_dem_instructions_include_correlations`

- biased noise
  - 結論: 固定の biased-noise 前提はない。任意 weight / probability を受け取る generic 設計である。
  - 根拠:
    - `src/pymatching/matching.py` / `Matching.load_from_check_matrix`, `Matching.add_edge`, `Matching.add_boundary_edge`
  - ただし:
    - repo 付属例として `Y` に偏らせた test はある。
  - 根拠:
    - `tests/matching/decode_test.py` / `test_correlated_matching_handles_single_detector_components`

### 2.2 ベンチマーク / シミュレーションスクリプトが実際に使うノイズモデル

#### 2.2.1 rotated surface code benchmarks

- 結論: `benchmarks/surface_codes` と `docs/getting-started.ipynb` の rotated surface code 評価は `circuit-level depolarising noise` を前提にしている。
  - 明示的記述:
    - benchmark README が `surface code circuits with circuit-level noise` と書く。
    - 使っている Stim 生成パラメータは `after_clifford_depolarization`, `before_round_data_depolarization`, `before_measure_flip_probability`, `after_reset_flip_probability`。
  - 根拠:
    - `benchmarks/surface_codes/README.md` / `save_benchmark_circuit`, `time_surface_code_circuit`
    - `docs/getting-started.ipynb` / `surface_code_circuit`, `get_logical_error_rate_pymatching`
    - `docs/toric-code-example.ipynb` / surface-code circuit-level section

- measurement error
  - 結論: `あり`
  - 根拠:
    - `benchmarks/surface_codes/README.md` / `before_measure_flip_probability=p`
    - `docs/getting-started.ipynb` / surface-code circuit construction

- データ誤り
  - 結論: `あり`
  - 根拠:
    - `benchmarks/surface_codes/README.md` / `before_round_data_depolarization=p`, `after_clifford_depolarization=p`

- 相関誤り / Y 誤り
  - 結論: DEM 実例には separator `^` を含む decomposed error が現れ、correlated matching の評価データも同梱される。
  - 根拠:
    - `data/surface_code_rotated_memory_x_13_0.01.dem`
    - `tests/matching/decode_test.py` / `test_surface_code_solution_weights_with_correlations`
  - 具体例:
    - `data/surface_code_rotated_memory_x_13_0.01.dem` 冒頭には `error(...) D1 D7 ^ D90` や `error(...) D1 L0 ^ D84 L0` があり、単一 edge ではなく decomposition 済み相関由来 instruction を含む。

- biased noise
  - 結論: benchmark README / docs の surface-code benchmark 本体は `biased noise` ではなく、対称な `p` を各 noise channel に与える設定である。
  - 根拠:
    - `benchmarks/surface_codes/README.md`
    - `docs/getting-started.ipynb`

#### 2.2.2 toric code notebook

- perfect syndrome section
  - 結論: `independent data-only noise`, `perfect syndrome`, `Z errors on qubits` を使っている。
  - 明示的記述:
    - notebook が independent noise model with perfect syndrome measurements と書く。
    - 各 qubit が independently `Z` error を受ける説明がある。
  - 根拠:
    - `docs/toric-code-example.ipynb` / `num_decoding_failures_via_physical_frame_changes`, `num_decoding_failures`
  - 分類:
    - `code capacity` というラベル自体は notebook に明記されていないため、そう呼ぶなら `推論`。

- noisy syndrome section
  - 結論: `phenomenological noise` を使っている。
  - 明示的記述:
    - notebook が phenomenological error model と明記。
    - data-qubit error probability `p` と measurement error probability `q` を独立にサンプル。
  - 根拠:
    - `docs/toric-code-example.ipynb` / `num_decoding_failures_noisy_syndromes`

- circuit-level section
  - 結論: surface-code 例に切り替えて `circuit-level noise` を評価する。
  - 根拠:
    - `docs/toric-code-example.ipynb` / circuit-level section

### 2.3 観点別まとめ

| 観点 | デコーダ前提 | repo 付属の実評価 |
|---|---|---|
| code capacity | 明示: check-matrix perfect-syndrome usage は可能 | 明示: toric notebook perfect-syndrome section |
| phenomenological | 明示: `repetitions` + `measurement_error_probabilities` + difference syndrome | 明示: toric notebook noisy syndrome section |
| circuit-level | 明示: DEM / Stim circuit を graphlike or decomposed graphlike DEM に変換して decode | 明示: benchmarks, README, getting-started notebook, surface-code DEM test data |
| measurement error | 明示: check-matrix の timelike edges と DEM detectors で対応 | 明示: benchmark circuits の `before_measure_flip_probability`, toric phenom notebook |
| data errors only | 明示: check-matrix perfect-syndrome path | 明示: toric notebook perfect-syndrome section |
| 相関誤り / Y error | 明示: correlated matching で decomposition 後の edge correlations を扱う | 明示: correlated test data, correlated decode tests |
| biased noise | 明示: 固定前提なし。任意 weight/probability を入力可 | 明示: Y-only test はあるが benchmark 主体ではない |

## 3. デコードアルゴリズムの概要

### 3.1 実装されているアルゴリズム

- `標準 MWPM`
  - 結論: repo の主 decoder は `Minimum Weight Perfect Matching`。
  - 根拠:
    - `README.md`
    - `src/pymatching/matching.py` / `class Matching`
    - `src/pymatching/sparse_blossom/matcher/mwpm.h` / `struct Mwpm`
    - `src/pymatching/sparse_blossom/driver/mwpm_decoding.cc` / `decode_detection_events`

- `sparse blossom`
  - 結論: MWPM の実装は README で `sparse blossom` と呼ばれている。
  - 根拠:
    - `README.md`
    - `src/pymatching/sparse_blossom/`

- `two-pass correlated matching`
  - 結論: correlated decoding は `MWPM を 2 回走らせる` 実装。
  - 明示的記述:
    - README が first pass で edges を予測し、reweight 後に second pass を行うと説明。
    - C++ 実装でも `decode_detection_events_to_edges` → `reweight_for_edges` → `decode_detection_events` の順。
  - 根拠:
    - `README.md`
    - `src/pymatching/sparse_blossom/driver/mwpm_decoding.cc` / `decode_detection_events`, `decode_detection_events_to_edges_with_edge_correlations`

- `CNN / GNN / neural decoder`
  - 結論: `なし`
  - 根拠:
    - `src/pymatching/`
    - `benchmarks/`
    - `docs/`
    - repo 全体のコード探索

### 3.2 パイプライン別の処理

#### 3.2.1 check-matrix パイプライン

- 構成
  - `Matching.load_from_check_matrix` / `Matching.from_check_matrix` が binary check matrix から matching graph を構成し、その上で MWPM を実行する。
  - 根拠:
    - `src/pymatching/matching.py` / `Matching.load_from_check_matrix`, `Matching.from_check_matrix`

- 物理誤りから syndrome graph への落とし込み
  - 各 column が 1 または 2 個の parity check を反転させる graphlike error mechanism として扱われ、matching graph の edge になる。
  - 根拠:
    - `README.md`
    - `src/pymatching/matching.py` / `Matching.load_from_check_matrix`

- repeated rounds
  - `repetitions>1` では timelike edges が追加され、3D matching graph 相当の構成になる。
  - 根拠:
    - `src/pymatching/matching.py` / `Matching.load_from_check_matrix`, `Matching.decode`

#### 3.2.2 DEM / Stim パイプライン

- 構成
  - `Matching.from_detector_error_model` / `Matching.from_stim_circuit` が `stim.DetectorErrorModel` から user graph を作り、その graph を MWPM に変換して decode する。
  - 根拠:
    - `src/pymatching/matching.py` / `Matching.from_detector_error_model`, `Matching.from_stim_circuit`
    - `src/pymatching/sparse_blossom/driver/user_graph.cc` / `detector_error_model_to_user_graph`
    - `src/pymatching/sparse_blossom/driver/mwpm_decoding.h` / `detector_error_model_to_mwpm`

- 物理誤りから syndrome graph への落とし込み
  - DEM の `graphlike error` は detector set に応じて edge / boundary edge になる。
  - 根拠:
    - `src/pymatching/matching.py` / `Matching.from_detector_error_model`
    - `src/pymatching/sparse_blossom/driver/user_graph.cc` / `handle_dem_instruction`

- decomposition 済み hyperedge
  - separator `^` を含む decomposed DEM instruction は component ごとの edge として展開される。
  - correlations enabled では、それらの component 間の joint probability から implied weight を作る。
  - 根拠:
    - `src/pymatching/sparse_blossom/driver/user_graph.h` / `iter_detector_error_model_edges`, `iter_dem_instructions_include_correlations`
    - `src/pymatching/sparse_blossom/driver/user_graph.cc` / `populate_implied_edge_weights`

- parallel edges
  - DEM 由来 parallel edges は `independent` 前提で merge される。
  - 根拠:
    - `src/pymatching/matching.py` / `Matching.from_detector_error_model`
    - `src/pymatching/sparse_blossom/driver/user_graph.cc` / `handle_dem_instruction`, `merge_edge_or_boundary_edge`

### 3.3 X/Z バリアントの扱い

- `XXZZ / XZZX`
  - 結論: `未確認`
  - 理由:
    - repo 内に variant 名や専用マッピング実装が見つからない。

- rotated surface-code benchmark における X/Z basis
  - 明示的記述:
    - benchmark README は `both X basis and Z basis measurements` を含む detector error model を decode していると書く。
  - 根拠:
    - `benchmarks/surface_codes/README.md`

- 実装上 separate X/Z matching かどうか
  - 明示的記述:
    - API / 実装は detector graph を generic に decode するだけで、X matching graph と Z matching graph を別オブジェクトに分ける surface-code 専用 API は見当たらない。
  - 根拠:
    - `src/pymatching/matching.py` / `class Matching`
    - `src/pymatching/sparse_blossom/driver/user_graph.cc` / `detector_error_model_to_user_graph`
  - 推論:
    - X/Z が独立な 2 グラフになるか、1 つの連結 graph になるかは `DEM の topology` に依存し、repo が variant 名から分岐しているわけではない。

### 3.4 高速化の工夫

- ユーザ要件に従い `調査対象外` とする。

## 4. 入出力インターフェースとデコードの運用形態

### 4.1 Python API: `Matching.decode`

#### 4.1.1 入力

- 型
  - `numpy.ndarray` または配列化可能な 1D/2D binary data。
  - 根拠:
    - `src/pymatching/matching.py` / `_syndrome_array_to_detection_events`, `Matching.decode`

- syndrome の定義
  - check-matrix, no repetitions:
    - 各ビットは parity check / detector node の syndrome bit。
    - 根拠:
      - `src/pymatching/matching.py` / `Matching.decode`
  - check-matrix, `repetitions>1`:
    - 2D 入力 `z[i,j]` は stabilizer `i` の連続ラウンド差分 `difference syndrome`。
    - 根拠:
      - `src/pymatching/matching.py` / `Matching.decode`
  - DEM / Stim:
    - README は `full syndrome (detector measurements)` と書くが、C++ 実装と関数名は `detection_events` を前提にしている。
    - 根拠:
      - `README.md`
      - `src/pymatching/matching.py` / `_syndrome_array_to_detection_events`
      - `src/pymatching/sparse_blossom/driver/mwpm_decoding.h` / `decode_detection_events`
    - 推論:
      - DEM/Stim パイプラインでは raw stabilizer 値そのものではなく `detector event bit array` を入力としていると読むのが実装整合的。

- 最終ラウンドの readout 再構成
  - check-matrix phenomenological example:
    - notebook は final round を measurement-error-free にするか、post-processing で stabilizer を正確に求める前提を明記。
    - 根拠:
      - `docs/toric-code-example.ipynb` / `num_decoding_failures_noisy_syndromes`
  - DEM / Stim:
    - PyMatching 自身が final round readout から再構成するコードは見当たらず、Stim から渡された detector events をそのまま decode する。
    - 根拠:
      - `src/pymatching/matching.py` / `Matching.from_stim_circuit`, `Matching.decode_batch`
      - `README.md`

#### 4.1.2 出力

- `decode`
  - 出力は `fault_ids` の binary vector。
  - 根拠:
    - `src/pymatching/matching.py` / `Matching.decode`

- `fault_ids` の意味
  - manual graph / check matrix default:
    - physical Pauli error / physical frame change を表せる。
  - check matrix + `faults_matrix`:
    - logical observables を直接表せる。
  - DEM / Stim:
    - `fault_ids` は DEM の `logical_observable` indices になる。
  - 根拠:
    - `src/pymatching/matching.py` / `Matching.add_edge`, `Matching.add_boundary_edge`, `Matching.load_from_check_matrix`, `Matching.from_detector_error_model`

#### 4.1.3 運用形態

- `physical correction / Pauli frame update / logical readout-only`
  - check-matrix path:
    - physical frame も logical observable prediction も両方可能。
  - DEM / Stim path:
    - shipped examples は logical observable prediction が中心。
  - 根拠:
    - `README.md`
    - `docs/toric-code-example.ipynb` / `num_decoding_failures_via_physical_frame_changes`, `num_decoding_failures`
    - `tests/matching/decode_test.py` / `test_surface_code_solution_weights`

- `active correction support`
  - 結論: `未確認`
  - 理由:
    - decoder は correction vector / edge set / logical observable bits を返すが、quantum circuit へ即時フィードバックする制御 API は repo に見当たらない。

### 4.2 Python API: `Matching.decode_batch`

- 入力データ
  - `np.uint8` 2D array。
  - shape は `(num_shots, syndrome_length)`、bit-packed も可。
  - 根拠:
    - `src/pymatching/matching.py` / `Matching.decode_batch`
    - `src/pymatching/sparse_blossom/driver/user_graph.pybind.cc` / `decode_batch`

- 出力データ
  - `predictions.shape = (num_shots, self.num_fault_ids)`。
  - bit-packed prediction も可。
  - `return_weights=True` なら解 weight も返す。
  - 根拠:
    - `src/pymatching/matching.py` / `Matching.decode_batch`

- shipped surface-code pipeline での意味
  - README / notebook の DEM 例では `predicted_observables` を返す logical readout decoder として使われる。
  - 根拠:
    - `README.md`
    - `docs/getting-started.ipynb` / `get_logical_error_rate_pymatching`

### 4.3 `decode_to_edges_array` / `decode_to_matched_dets_array`

- `decode_to_edges_array`
  - full correction に近い edge-level 出力。返り値は detector node pair、boundary は `-1`。
  - 根拠:
    - `src/pymatching/matching.py` / `decode_to_edges_array`
    - `src/pymatching/sparse_blossom/driver/mwpm_decoding.h` / `decode_detection_events_to_edges`

- `decode_to_matched_dets_array` / `decode_to_matched_dets_dict`
  - detection events 間の matching pair だけを返し、full edge path ではない。
  - 根拠:
    - `src/pymatching/matching.py` / `decode_to_matched_dets_array`, `decode_to_matched_dets_dict`
    - `src/pymatching/sparse_blossom/driver/mwpm_decoding.h` / `decode_detection_events_to_match_edges`

### 4.4 CLI

- `predict`
  - 入力:
    - `--dem` と shot file (`b8`, `dets` など)。
    - `--in_includes_appended_observables` 指定可。
  - 出力:
    - `logical observable bits`。writer は `begin_result_type('L')`。
  - 根拠:
    - `src/pymatching/sparse_blossom/driver/namespaced_main.cc` / `main_predict`

- `count_mistakes`
  - 入力:
    - detector shots と observables。
  - 出力:
    - `num_mistakes / num_shots`。
  - 根拠:
    - `src/pymatching/sparse_blossom/driver/namespaced_main.cc` / `main_count_mistakes`

## 5. Neural network 系アルゴリズムの対応

### 5.1 結論

- 結論: `該当なし`
  - repo 内に neural-network decoder の training / inference 実装は見当たらない。
  - 根拠:
    - `src/pymatching/`
    - `benchmarks/`
    - `docs/`
    - repo 全体のコード探索

### 5.2 training / inference / synthetic data generation

- `training`
  - なし
- `inference`
  - なし
- `synthetic training data generation`
  - なし
- 補足
  - Stim や notebook は logical-error evaluation 用の synthetic data は生成するが、これは neural training dataset 生成ではない。
  - 根拠:
    - `docs/getting-started.ipynb`
    - `docs/toric-code-example.ipynb`
    - `benchmarks/surface_codes/README.md`

## 6. ベンチマークの評価内容

### 6.1 `benchmarks/surface_codes` に保存されている結果

- 何を評価しているか
  - 結論: 主に `decode speed`。
  - 明示的記述:
    - `time_surface_code_circuit` は `matching.decode_batch(shots)` の wall-clock time から `microseconds_per_shot` を計算する。
    - `pymatching_v2.csv` は `d,p,microseconds` を保存している。
  - 根拠:
    - `benchmarks/surface_codes/README.md` / `time_surface_code_circuit`
    - `benchmarks/surface_codes/.../pymatching_v2.csv`

- logical memory 実験か
  - 明示的記述:
    - circuit は `surface_code:rotated_memory_x`。
    - benchmark README は `measuring the X logical observable` と書く。
  - 根拠:
    - `benchmarks/surface_codes/README.md`
  - 判定:
    - `surface-code logical memory circuit の decode timing benchmark` とみなせる。

- 評価前提
  - `rounds = distance`
  - `after_clifford_depolarization = p`
  - `before_round_data_depolarization = p`
  - `before_measure_flip_probability = p`
  - `after_reset_flip_probability = p`
  - detector set は `both X and Z basis measurements`
  - ただし logical error rate への直接寄与は README 上 `X logical observable` に限定
  - 根拠:
    - `benchmarks/surface_codes/README.md`

### 6.2 notebook 内の logical-error evaluation

#### 6.2.1 `docs/getting-started.ipynb`

- 何を評価しているか
  - 結論: `logical error rate` と threshold plot。
  - 根拠:
    - `docs/getting-started.ipynb` / `get_logical_error_rate_pymatching`, threshold plot section

- 前提条件
  - circuit:
    - `surface_code:rotated_memory_x`
  - rounds:
    - `rounds=d`
  - noise:
    - circuit-level depolarising + measurement/reset/data noise all at rate `p`
  - evaluation:
    - `predicted_obs != obs` の shot 数を logical error と数える
  - 根拠:
    - `docs/getting-started.ipynb` / `surface_code_circuit`, `get_logical_error_rate_pymatching`

- correlated decoder 比較
  - 結論: uncorrelated と correlated の threshold を比較する notebook section がある。
  - 根拠:
    - `docs/getting-started.ipynb` / correlated threshold comparison section

#### 6.2.2 `docs/toric-code-example.ipynb`

- perfect syndrome section
  - 何を評価しているか
    - toric code の logical error count / threshold。
  - 評価対象の偏り
    - `X stabilisers` と `X logical operators` に対し、qubit `Z error` を入れて decode。
  - 根拠:
    - `docs/toric-code-example.ipynb` / `toric_code_x_stabilisers`, `toric_code_x_logicals`, `num_decoding_failures`

- noisy syndrome section
  - 何を評価しているか
    - phenomenological noise 下の logical error count / threshold。
  - 前提条件
    - repetitions あり
    - difference syndrome
    - final round perfect measurements
  - 根拠:
    - `docs/toric-code-example.ipynb` / `num_decoding_failures_noisy_syndromes`

- circuit-level section
  - 何を評価しているか
    - surface code の logical error rate vs physical error rate。
  - 明示的記述:
    - notebook は `threshold of around 0.7% for circuit-level depolarising noise in the surface code` と述べる。
  - 根拠:
    - `docs/toric-code-example.ipynb` / circuit-level section

### 6.3 まとめ

- 保存済み benchmark artifact:
  - 主に `timing benchmark`
- docs notebook:
  - `logical error rate / threshold evaluation`
- surface-code shipped evaluation の主対象:
  - `rotated_memory_x` logical memory task
- toric notebook の偏り:
  - `X stabiliser / X logical` セクタに偏る
- active correction benchmark:
  - `未確認`

---

## 最終所見

- この repo は surface code を `専用に実装した repo` ではなく、`graphlike Tanner graph / DEM を受け取る汎用 MWPM decoder` として surface-code 系問題を処理している。
- repo 内で明示的に確認できる surface-code family は主に `Stim の surface_code:rotated_memory_x` と `toric code example`。
- ノイズモデルは `code-capacity-like`, `phenomenological`, `circuit-level` の 3 種が docs / API 上で確認できる。
- neural decoder, lattice surgery 専用実装, XZZX / XXZZ 専用実装は `未確認`。
