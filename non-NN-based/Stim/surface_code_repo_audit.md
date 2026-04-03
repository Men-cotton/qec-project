# Stim Repository Audit: Surface Code / Decoder / Benchmark

この文書は、`/home/mencotton/qec-project/Stim` を対象に、指定観点ごとに段階的に追記した調査メモである。

## 調査ルール

- 各主張について、根拠ファイルと関数/クラス名を付す。
- `明示的記述` と `推論` を分ける。
- 証拠が見つからない場合は `未確認` と書く。
- `未実装` と `設計上の対象外` を区別する。

---

## 1. Surface code の対応状況と実装範囲

### 1-1. 結論

- この repo は surface code の QEC に対応している。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `stim::generate_surface_code_circuit`, `_generate_rotated_surface_code_circuit`, `_generate_unrotated_surface_code_circuit`
  ファイル [`src/stim/cmd/command_gen.cc`](/home/mencotton/qec-project/Stim/src/stim/cmd/command_gen.cc) / file-level CLI help (`--code`, `--task`)

- surface-code 関連の主対象ディレクトリは少なくとも以下。
  - 回路生成: [`src/stim/gen`](/home/mencotton/qec-project/Stim/src/stim/gen) / 関数 `stim::generate_surface_code_circuit`
  - CLI 露出: [`src/stim/cmd`](/home/mencotton/qec-project/Stim/src/stim/cmd) / file-level command handlers
  - サンプリング/DET 変換/DEM: [`src/stim/simulators`](/home/mencotton/qec-project/Stim/src/stim/simulators), [`src/stim/util_top`](/home/mencotton/qec-project/Stim/src/stim/util_top) / クラス `CompiledDetectorSampler` ほか
  - ベンチマーク/デコード運用例: [`glue/sample/src/sinter`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter) / モジュール-level 実装
  - lattice surgery 合成: [`glue/lattice_surgery`](/home/mencotton/qec-project/Stim/glue/lattice_surgery) / クラス `LatticeSurgerySynthesizer`, `LatticeSurgerySolution`

### 1-2. 対応している code family の明示

- `surface_code` 生成器で明示的に提供されている task は `rotated_memory_x`, `rotated_memory_z`, `unrotated_memory_x`, `unrotated_memory_z` の 4 つのみ。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `stim::generate_surface_code_circuit`
  ファイル [`src/stim/cmd/command_gen.cc`](/home/mencotton/qec-project/Stim/src/stim/cmd/command_gen.cc) / file-level CLI help

- rotated variant は X stabilizer 測定点と Z stabilizer 測定点を別集合で持つ CSS 型 surface code として実装されている。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_generate_rotated_surface_code_circuit`, `_finish_surface_code_circuit`

- unrotated variant も X stabilizer 測定点と Z stabilizer 測定点を別集合で持つ CSS 型 surface code として実装されている。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_generate_unrotated_surface_code_circuit`, `_finish_surface_code_circuit`

- `XZZX` surface code, `XXZZ` rotated planar などの名称・task・専用生成器は repo 内で確認できなかったため、対応は `未確認` ではなく、少なくとも回路生成器としての明示的サポートは `未実装` と判断する。
  根拠:
  ファイル全体検索で `XZZX`, `XXZZ` に surface-code 生成器や task 名としての一致が見当たらない
  検索対象: `src`, `glue`, `doc`, `README.md`
  関数/クラス: 該当なし

### 1-3. 実装制約

- 単一 logical qubit の memory 実験に限定されている。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_finish_surface_code_circuit`
  観察点:
  `OBSERVABLE_INCLUDE` は常に index `0` のみを追加している。
  `x_observable`, `z_observable` も 1 本の logical path として構築されている。

- patch shape は square patch のみが明示的実装で、rectangular patch パラメータはない。
  根拠:
  ファイル [`src/stim/gen/circuit_gen_params.h`](/home/mencotton/qec-project/Stim/src/stim/gen/circuit_gen_params.h) / 構造体 `CircuitGenParameters`
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_generate_rotated_surface_code_circuit`, `_generate_unrotated_surface_code_circuit`
  観察点:
  幾何は単一整数 `distance` のみから生成され、独立な width/height を受け取らない。

- 境界は open boundary の planar patch と読むのが妥当。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_generate_rotated_surface_code_circuit`, `_generate_unrotated_surface_code_circuit`
  明示的記述:
  境界上で測定子配置を間引く条件分岐がある。
  推論:
  トーリックコードのような周期境界を作る処理は見当たらず、境界つき平面パッチ実装と読むのが自然。

- odd distance 制約は surface code 生成器には見当たらない。`distance >= 2` のみを検証している。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_finish_surface_code_circuit`
  ファイル [`src/stim/gen/gen_surface_code.test.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.test.cc) / test `rotated_surface_code_hard_coded_comparison`
  観察点:
  `params.distance < 2` のみ例外。
  test で rotated code の `distance = 4` が通っている。

- repeated syndrome rounds に対応する。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_finish_surface_code_circuit`
  観察点:
  `body * (params.rounds - 1)` により繰り返し測定ラウンドを構成する。

- measurement error を含む fault-tolerant memory circuit を生成できる。
  根拠:
  ファイル [`src/stim/gen/circuit_gen_params.h`](/home/mencotton/qec-project/Stim/src/stim/gen/circuit_gen_params.h) / 構造体 `CircuitGenParameters`
  ファイル [`src/stim/gen/circuit_gen_params.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/circuit_gen_params.cc) / 関数 `append_measure`, `append_measure_reset`
  観察点:
  `before_measure_flip_probability` に応じて測定直前に反対基底の Pauli エラーを注入する。

- active correction を回路内部で実行する仕組みは、surface code 生成器では確認できない。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_finish_surface_code_circuit`
  明示的記述:
  出力回路は detector と observable annotation を含む memory circuit。
  推論:
  デコーダ出力を使って回路途中に補正ゲートやフィードバックを適用する経路はこの生成器には存在しないため、用途は logical readout / Pauli frame 評価寄り。

- lattice surgery は別系統の Python glue として存在するが、surface-code memory generator の task として統合はされていない。
  根拠:
  ファイル [`glue/lattice_surgery/README.md`](/home/mencotton/qec-project/Stim/glue/lattice_surgery/README.md) / file-level README
  ファイル [`glue/lattice_surgery/lassynth/__init__.py`](/home/mencotton/qec-project/Stim/glue/lattice_surgery/lassynth/__init__.py) / エクスポート `LatticeSurgerySynthesizer`, `LatticeSurgerySolution`
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `stim::generate_surface_code_circuit`
  推論:
  repo 全体としては surface-code lattice surgery を扱うコードがあるが、Stim 本体の `surface_code:*` generator とは別実装である。

### 1-4. Capability Matrix

| code family | patch shape | single/multi logical qubit | boundaries | odd distance restriction | repeated syndrome rounds | measurement error support | active correction support | lattice surgery | benchmark scripts | neural decoder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `surface_code:rotated_memory_x` / `rotated_memory_z` | square rotated planar patch | single logical qubit | open boundary planar patch | なし。`distance >= 2` | あり | あり | 明示的にはなし | なし | あり | なし |
| `surface_code:unrotated_memory_x` / `unrotated_memory_z` | square unrotated planar patch | single logical qubit | open boundary planar patch | なし。`distance >= 2` | あり | あり | 明示的にはなし | なし | `未確認` | なし |
| `glue/lattice_surgery` (LaSsynth) | patch というより 3D subroutine volume | 複数 port/stabilizer を扱うため multi-logical を含みうる | `未確認` | `未確認` | N/A | N/A | N/A | あり | `未確認` | なし |

### 1-5. Matrix 各セルの根拠

- `rotated_memory_*`:
  根拠ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_generate_rotated_surface_code_circuit`, `_finish_surface_code_circuit`
  補助根拠 [`src/stim/gen/circuit_gen_params.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/circuit_gen_params.cc) / 関数 `append_measure`, `append_measure_reset`
  benchmark 根拠 [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md) / file-level example using `surface_code:rotated_memory_x`

- `unrotated_memory_*`:
  根拠ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_generate_unrotated_surface_code_circuit`, `_finish_surface_code_circuit`
  補助根拠 [`src/stim/gen/gen_surface_code.test.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.test.cc) / test `unrotated_surface_code_hard_coded_comparison`

- `glue/lattice_surgery`:
  根拠ファイル [`glue/lattice_surgery/README.md`](/home/mencotton/qec-project/Stim/glue/lattice_surgery/README.md) / file-level README
  根拠ファイル [`glue/lattice_surgery/lassynth/__init__.py`](/home/mencotton/qec-project/Stim/glue/lattice_surgery/lassynth/__init__.py) / exported classes

### 1-6. 未対応理由の切り分け

- `XZZX` / `XXZZ` など非 CSS あるいは別 stabilizer 配置の surface code:
  判定: `未実装`
  理由:
  生成器は X 測定子集合と Z 測定子集合を分けて組み、回路も X/Z basis stabilizer cycle を前提にしている。
  根拠ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_finish_surface_code_circuit`, `_generate_rotated_surface_code_circuit`, `_generate_unrotated_surface_code_circuit`

- rectangular planar patch:
  判定: `未実装`
  理由:
  生成パラメータが単一 `distance` のみで、縦横別指定のアーキテクチャになっていない。
  根拠ファイル [`src/stim/gen/circuit_gen_params.h`](/home/mencotton/qec-project/Stim/src/stim/gen/circuit_gen_params.h) / 構造体 `CircuitGenParameters`

- multi-logical memory patch:
  判定: `設計上の対象外` と読むのが妥当
  理由:
  surface code generator の task が memory 実験 4 種に固定され、observable も 1 本だけ構成する。
  根拠ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `stim::generate_surface_code_circuit`, `_finish_surface_code_circuit`

- lattice surgery:
  判定: repo 全体では `別系統で対応`
  理由:
  `glue/lattice_surgery` は SAT/SMT ベースの subroutine synthesizer であり、Stim 本体の memory circuit generator の拡張 task ではない。
  根拠ファイル [`glue/lattice_surgery/README.md`](/home/mencotton/qec-project/Stim/glue/lattice_surgery/README.md) / file-level README

---

## 2. 対象ノイズモデル

### 2-1. 結論の要約

- surface-code 回路生成器自体は、`DEPOLARIZE1`, `DEPOLARIZE2`, 測定直前の anti-basis Pauli flip, reset 直後の anti-basis Pauli flip を組み合わせた `Pauli-only` ノイズ注入に対応する。
  根拠:
  ファイル [`src/stim/gen/circuit_gen_params.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/circuit_gen_params.cc) / 関数 `append_begin_round_tick`, `append_unitary_1`, `append_unitary_2`, `append_reset`, `append_measure`, `append_measure_reset`
  ファイル [`doc/stim.pyi`](/home/mencotton/qec-project/Stim/doc/stim.pyi) / 関数 `stim.Circuit.generated`

- `sinter` の既定フローは、回路から `DetectorErrorModel` を自動生成するとき `decompose_errors=True, approximate_disjoint_errors=True` をまず試す。
  根拠:
  ファイル [`glue/sample/src/sinter/_collection/_collection_worker_state.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_collection/_collection_worker_state.py) / 関数 `_fill_in_task`

- よって benchmark / decoding の標準経路は、回路レベル Pauli ノイズから導いた DEM を入力にして、外部デコーダに observable flip 予測をさせる構成である。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding.py) / 関数 `sample_decode`, `_sample_decode_helper_using_memory`, `_sample_decode_helper_using_disk`

### 2-2. デコーダ実装が仮定するノイズモデル

#### 2-2-a. DEM 共通前提

- `DetectorErrorModel` は「独立な error mechanism の列」として定義されている。
  根拠:
  ファイル [`doc/file_format_dem_detector_error_model.md`](/home/mencotton/qec-project/Stim/doc/file_format_dem_detector_error_model.md) / file-level format semantics

- `stim.Circuit.detector_error_model()` は、互いに独立でない disjoint component を含むノイズチャネルについて、`approximate_disjoint_errors=True` を使うと独立事象として近似する。
  根拠:
  ファイル [`src/stim/circuit/circuit.pybind.cc`](/home/mencotton/qec-project/Stim/src/stim/circuit/circuit.pybind.cc) / pybind method `detector_error_model`

- `decompose_errors=True` は、複合誤りを graphlike component に分解する提案を `^` separator で DEM に埋め込む。
  根拠:
  ファイル [`src/stim/circuit/circuit.pybind.cc`](/home/mencotton/qec-project/Stim/src/stim/circuit/circuit.pybind.cc) / pybind method `detector_error_model`
  ファイル [`src/stim/cmd/command_analyze_errors.cc`](/home/mencotton/qec-project/Stim/src/stim/cmd/command_analyze_errors.cc) / CLI help for `--decompose_errors`

#### 2-2-b. PyMatching

- `pymatching` 実装は `pymatching.Matching.from_detector_error_model(dem)` にそのまま DEM を渡すだけで、この repo 側では独自の重み再計算や syndrome graph 再構成はしていない。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_pymatching.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_pymatching.py) / クラス `PyMatchingDecoder`, `PyMatchingCompiledDecoder`

- 明示的記述:
  PyMatching 側の詳細ロジックは repo 外部依存であり、この repo 内コードからは MWPM の細部までは確認できない。

- 推論:
  この repo から確認できる範囲では、PyMatching 経路は `graphlike DEM` を前提にする decoding path と整理するのが妥当。
  根拠:
  ファイル [`glue/sample/src/sinter/_collection/_collection_worker_state.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_collection/_collection_worker_state.py) / 関数 `_fill_in_task`
  ファイル [`src/stim/cmd/command_analyze_errors.cc`](/home/mencotton/qec-project/Stim/src/stim/cmd/command_analyze_errors.cc) / `--decompose_errors` help

#### 2-2-c. fusion_blossom

- `fusion_blossom` 実装は detector pair もしくは detector-boundary の edge に変換できる誤りのみを受け付け、`len(dets) > 2` で `NotImplementedError` を投げる。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py) / 関数 `detector_error_model_to_fusion_blossom_solver_and_fault_masks`

- separator `^` で分解された component は「独立誤り」として個別 edge 化する。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py) / 関数 `iter_flatten_model`

- edge weight は `log((1 - p) / p)` で計算しており、`p > 0.5` は負重みを避けるため `0.5` に丸める。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py) / 関数 `detector_error_model_to_fusion_blossom_solver_and_fault_masks`

- 推論:
  この経路は phenomenological 専用ではなく、`graphlike に分解済みで独立 edge 近似できる circuit-level DEM` を扱う MWPM 系 path である。

#### 2-2-d. MWPF / hypergraph_union_find

- `mw_parity_factor` は detector 集合サイズに上限を設けず hyperedge を構築する。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_mwpf.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_mwpf.py) / 関数 `detector_error_model_to_mwpf_solver_and_fault_masks`

- `hypergraph_union_find` は `HyperUFDecoder` として `MwpfDecoder` の solver class を変えた alias である。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_all_built_in_decoders.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_all_built_in_decoders.py) / dict `BUILT_IN_DECODERS`
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_mwpf.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_mwpf.py) / クラス `HyperUFDecoder`

- edge/hyperedge weight は同じく `log((1 - p) / p)` で計算し、`p > 0.5` は `0.5` に丸める。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_mwpf.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_mwpf.py) / 関数 `detector_error_model_to_mwpf_solver_and_fault_masks`

- 同一 detector set の hyperedge は確率合成して deduplicate し、logical fault mask は「より尤もらしい方」の mask を残す。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_mwpf.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_mwpf.py) / 関数 `deduplicate_hyperedges`

- 推論:
  MWPF 系は graphlike 制約を外した `hypergraph DEM` 前提の decoding path と言える。ただしここでも誤りの基本単位は DEM に列挙された独立 mechanism である。

#### 2-2-e. vacuous

- `vacuous` decoder は常に observable flip なしを予測し、ノイズモデル仮定は実質持たない。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_vacuous.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_vacuous.py) / クラス `VacuousDecoder`, `VacuousCompiledDecoder`

### 2-3. ベンチマーク / シミュレーションスクリプトが実際に使うノイズモデル

#### 2-3-a. 主要な README / notebook の benchmark

- `glue/sample/README.md` の Linux CLI 例は、rotated surface code memory に対し
  `after_clifford_depolarization = p`
  `after_reset_flip_probability = p`
  `before_measure_flip_probability = p`
  `before_round_data_depolarization = p`
  を全て有効化している。
  根拠:
  ファイル [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md) / file-level example

- `doc/getting_started.ipynb` の surface-code 収集例も同じ 4 パラメータを全部 `noise` に設定している。
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / notebook cell generating `surface_code_tasks`

- 上記 2 例は、データ誤りのみではなく、measurement error と reset error を含む `circuit-level Pauli noise` ベンチマークである。
  根拠:
  ファイル [`src/stim/gen/circuit_gen_params.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/circuit_gen_params.cc) / 関数 `append_measure`, `append_measure_reset`, `append_unitary_2`, `append_begin_round_tick`
  推論:
  `DEPOLARIZE2` により 2 量子ビットの相関 Pauli fault が入り、measurement/reset/idle fault も別パラメータで有効化されているため、単純な code-capacity でも単純な phenomenological でもない。

#### 2-3-b. テストコードでの surface-code decode

- `glue/sample/src/sinter/_decoding/_decoding_test.py` の surface-code test は `after_clifford_depolarization=0.001` のみを設定し、他の 3 パラメータは未設定でデフォルト 0 のままである。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_test.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_test.py) / test `test_decode_surface_code`

- この test は circuit-level の一部サブケースであり、measurement error を含まない。
  根拠:
  ファイル [`doc/stim.pyi`](/home/mencotton/qec-project/Stim/doc/stim.pyi) / 関数 `stim.Circuit.generated`
  推論:
  2 量子ビット Clifford 後 depolarization はあるため code-capacity ではないが、measurement/reset/idle error は入っていない。

### 2-4. 観点別整理

#### 2-4-a. measurement error の有無

- 回路生成器: `before_measure_flip_probability` により measurement error を注入可能。
  根拠:
  ファイル [`src/stim/gen/circuit_gen_params.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/circuit_gen_params.cc) / 関数 `append_measure`, `append_measure_reset`

- README / notebook benchmark: measurement error あり。
  根拠:
  ファイル [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md) / file-level example
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / notebook example

- `_decoding_test.py` surface-code test: measurement error なし。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_test.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_test.py) / test `test_decode_surface_code`

#### 2-4-b. データ誤りのみか

- repo の主要 surface-code benchmark 例はデータ誤りのみではない。
  根拠:
  ファイル [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md) / file-level example
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / notebook example

- データ誤りのみの surface-code benchmark script は `未確認`。
  根拠:
  repo 内検索で surface-code benchmark と明示された例の大半が all-four-noise または after-Clifford-only
  関数/クラス: 該当なし

#### 2-4-c. 相関誤りや Y 誤りの扱い

- 回路ノイズには `DEPOLARIZE2` があるため、2 量子ビット Pauli 相関誤りを含められる。
  根拠:
  ファイル [`src/stim/gen/circuit_gen_params.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/circuit_gen_params.cc) / 関数 `append_unitary_2`

- surface code の Y 誤りは、DEM 文書および `--decompose_errors` 文書で X/Z component へ分解される例として明示されている。
  根拠:
  ファイル [`doc/file_format_dem_detector_error_model.md`](/home/mencotton/qec-project/Stim/doc/file_format_dem_detector_error_model.md) / file-level DEM format docs
  ファイル [`src/stim/cmd/command_analyze_errors.cc`](/home/mencotton/qec-project/Stim/src/stim/cmd/command_analyze_errors.cc) / `--decompose_errors` help

- `fusion_blossom` は separator ごとに独立 edge として扱うため、相関を保持した joint decoding はしていない。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py) / 関数 `iter_flatten_model`

- `MWPF` は hyperedge を保持できるため、graphlike でない detector support を DEM 上では保持できる。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_mwpf.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_mwpf.py) / 関数 `detector_error_model_to_mwpf_solver_and_fault_masks`

#### 2-4-d. biased noise 前提の有無

- surface-code generator / benchmark で biased noise を前提にしたパラメータや専用 task は確認できない。
  根拠:
  ファイル [`src/stim/gen/circuit_gen_params.h`](/home/mencotton/qec-project/Stim/src/stim/gen/circuit_gen_params.h) / 構造体 `CircuitGenParameters`
  repo 内検索で surface-code generator と biased-noise を直接結ぶ実装は見当たらない

- `glue/crumble/crumble.html` に UI 文字列として `Biased (XZZX)` はあるが、surface-code generator / decoder benchmark 実装の根拠にはならない。
  根拠:
  ファイル [`glue/crumble/crumble.html`](/home/mencotton/qec-project/Stim/glue/crumble/crumble.html) / file-level UI text
  判定:
  調査対象の surface-code QEC 実装に対する証拠としては採用しない。

### 2-5. 前提モデル分類

- 回路生成器が作れるもの:
  `code capacity` 専用ではない。`before_round_data_depolarization` だけを使えば data-only に近い設定は可能だが、生成器の標準表現は `circuit-level Pauli noise` を表現できる。
  根拠:
  ファイル [`src/stim/gen/circuit_gen_params.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/circuit_gen_params.cc) / 関数群

- README / notebook benchmark の実際の設定:
  `circuit-level Pauli noise`
  根拠:
  ファイル [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md)
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb)

- PyMatching / fusion_blossom を選んだ場合の典型経路:
  `circuit-level Pauli noise` から導出した `graphlike DEM` を MWPM 系 decoder に渡す形。
  根拠:
  ファイル [`glue/sample/src/sinter/_collection/_collection_worker_state.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_collection/_collection_worker_state.py) / 関数 `_fill_in_task`
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_pymatching.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_pymatching.py) / `PyMatchingDecoder`
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py) / `FusionBlossomDecoder`

- `mw_parity_factor` / `hypergraph_union_find` 経路:
  `circuit-level Pauli noise` から導出した `hypergraph DEM` を直接扱いうる path。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_mwpf.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_mwpf.py)

---

## 3. デコードアルゴリズムの概要

### 3-1. shot-by-shot decoding pipeline として確認できたもの

#### Pipeline A: Stim sampling + PyMatching

- アルゴリズム種別: 外部 `pymatching` に委譲する MWPM 系 decoder。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_pymatching.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_pymatching.py) / クラス `PyMatchingDecoder`, `PyMatchingCompiledDecoder`
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / tutorial text + `count_logical_errors`

- この repo 内の役割:
  1. circuit を sampling して detection events / observable flips を得る
  2. circuit から DEM を作る
  3. DEM を `pymatching.Matching.from_detector_error_model` に渡す
  4. decoder が予測した observable flips と実測 observable flips を比較する
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / tutorial cells around `count_logical_errors`
  ファイル [`glue/sample/src/sinter/_decoding/_decoding.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding.py) / 関数 `sample_decode`

#### Pipeline B: Stim sampling + fusion_blossom

- アルゴリズム種別: MWPM 系 solver adapter。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py) / クラス `FusionBlossomDecoder`, `FusionBlossomCompiledDecoder`

- 実装上は DEM の各 graphlike component を edge にし、solver が返した subgraph に対応する fault mask の XOR を observable 予測へ変換する。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py) / 関数 `detector_error_model_to_fusion_blossom_solver_and_fault_masks`

#### Pipeline C: Stim sampling + MWPF

- アルゴリズム種別: Minimum-Weight Parity Factor。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_all_built_in_decoders.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_all_built_in_decoders.py) / comment on `mw_parity_factor`
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_mwpf.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_mwpf.py) / クラス `MwpfDecoder`, `MwpfCompiledDecoder`

- 実装上は DEM の各 error mechanism を hyperedge 化し、solver の subgraph に含まれる fault mask の XOR を observable 予測へ変換する。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_mwpf.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_mwpf.py) / 関数 `detector_error_model_to_mwpf_solver_and_fault_masks`

#### Pipeline D: Stim sampling + hypergraph union find

- アルゴリズム種別: weighted hypergraph union find。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_all_built_in_decoders.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_all_built_in_decoders.py) / comment on `hypergraph_union_find`
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_mwpf.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_mwpf.py) / クラス `HyperUFDecoder`

#### Pipeline E: vacuous baseline

- アルゴリズム種別: 常に `no flip` を返す baseline。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_vacuous.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_vacuous.py) / クラス `VacuousDecoder`

### 3-2. 実装上の共通点

- `sample_decode` / `StimThenDecodeSampler` は code family ごとに decoder を切り替えず、`Task.detector_error_model` と `Task.circuit` を generic に処理する。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding.py) / 関数 `sample_decode`
  ファイル [`glue/sample/src/sinter/_decoding/_stim_then_decode_sampler.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_stim_then_decode_sampler.py) / クラス `StimThenDecodeSampler`, `_CompiledStimThenDecodeSampler`

- 推論:
  rotated / unrotated の surface-code variant 差は decoder adapter 側ではなく、回路から生成される detection-event graph / DEM 側に押し込められている。

### 3-3. X/Z 取り扱いと variant 間共有の実態

- surface-code generator は X stabilizer 測定点と Z stabilizer 測定点を別集合で管理する CSS 実装である。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_generate_rotated_surface_code_circuit`, `_generate_unrotated_surface_code_circuit`

- `--decompose_errors` の説明は、CSS surface code では Y 誤りが X/Z の graphlike piece へ分解され、X/Z decoding graph の分離を保つことが重要だと明示している。
  根拠:
  ファイル [`src/stim/cmd/command_analyze_errors.cc`](/home/mencotton/qec-project/Stim/src/stim/cmd/command_analyze_errors.cc) / `--decompose_errors` help
  ファイル [`doc/file_format_dem_detector_error_model.md`](/home/mencotton/qec-project/Stim/doc/file_format_dem_detector_error_model.md) / DEM docs

- `fusion_blossom` は separator `^` ごとに独立 edge として扱う。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py) / 関数 `iter_flatten_model`

- `PyMatching` 経路でも repo 側には X-matching と Z-matching を別々に構築する明示コードはなく、DEM をそのまま外部ライブラリへ渡している。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_pymatching.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_pymatching.py) / クラス `PyMatchingDecoder`

- 推論:
  CSS の X/Z 分離は「decoder が basis-aware に枝分けする」のではなく、「DEM decomposition が symptom set を basis-compatible に整理する」ことで達成される設計である。

- `XXZZ` と `XZZX` を共通 decoder で扱う実装:
  `未確認`
  理由:
  そもそもその variant の surface-code generator / benchmark / decoder-specific branch が repo 内で見当たらない。

### 3-4. 非 decoder 系の関連アルゴリズム

- `stim.Circuit.shortest_graphlike_error` は graphlike logical error 探索であり、shot-by-shot syndrome decoding ではない。
  根拠:
  ファイル [`src/stim/circuit/circuit.pybind.cc`](/home/mencotton/qec-project/Stim/src/stim/circuit/circuit.pybind.cc) / method `shortest_graphlike_error`

- `stim.Circuit.search_for_undetectable_logical_errors` は hyper error を含む logical error search heuristic であり、これも shot-by-shot decoder ではない。
  根拠:
  ファイル [`src/stim/circuit/circuit.pybind.cc`](/home/mencotton/qec-project/Stim/src/stim/circuit/circuit.pybind.cc) / method `search_for_undetectable_logical_errors`

---

## 4. 入出力インターフェースとデコードの運用形態

### 4-1. 生成回路の syndrome 定義

- 生成される surface-code circuit の `DETECTOR` は raw stabilizer 値そのものではなく、measurement record の parity comparison で定義される detection event である。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_finish_surface_code_circuit`
  ファイル [`src/stim/cmd/command_m2d.cc`](/home/mencotton/qec-project/Stim/src/stim/cmd/command_m2d.cc) / subcommand help `m2d`
  ファイル [`src/stim/simulators/measurements_to_detection_events.pybind.cc`](/home/mencotton/qec-project/Stim/src/stim/simulators/measurements_to_detection_events.pybind.cc) / class `CompiledMeasurementsToDetectionEventsConverter`

- 初回ラウンドの detector:
  `head` では `chosen_basis_measure_coords` の measurement qubit の単独 measurement result を detector にしている。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_finish_surface_code_circuit`

- 中間ラウンドの detector:
  `body` では各 measurement qubit について「前ラウンドとの差分」になる 2 項 parity を detector にしている。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_finish_surface_code_circuit`

- 最終ラウンドの detector:
  `tail` では data qubit の最終 readout と直前の ancilla measurement を組み合わせて detector を再構成している。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_finish_surface_code_circuit`

- logical observable:
  `OBSERVABLE_INCLUDE(0)` に chosen-basis の data qubit readout parity を入れている。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / 関数 `_finish_surface_code_circuit`

### 4-2. 入力データ

#### Pipeline 共通 decoder API

- decoder compile 入力:
  `stim.DetectorErrorModel`
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_decoder_class.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_decoder_class.py) / クラス `Decoder`, `CompiledDecoder`

- shot decoding 入力:
  bit-packed detection event tensor
  `dtype=np.uint8`
  `shape=(num_shots, ceil(num_detectors / 8))`
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_decoder_class.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_decoder_class.py) / クラス `CompiledDecoder`

#### sampler 出力としての入力

- `stim.CompiledDetectorSampler.sample(..., separate_observables=True, bit_packed=True)` は
  `dets.dtype=uint8, shape=(shots, ceil(num_detectors/8))`
  `obs.dtype=uint8, shape=(shots, ceil(num_observables/8))`
  を返す。
  根拠:
  ファイル [`src/stim/py/compiled_detector_sampler.pybind.cc`](/home/mencotton/qec-project/Stim/src/stim/py/compiled_detector_sampler.pybind.cc) / method `sample`

- raw measurement から始める場合は `CompiledMeasurementsToDetectionEventsConverter` / `stim m2d` で detection event に変換する。
  根拠:
  ファイル [`src/stim/simulators/measurements_to_detection_events.pybind.cc`](/home/mencotton/qec-project/Stim/src/stim/simulators/measurements_to_detection_events.pybind.cc) / class `CompiledMeasurementsToDetectionEventsConverter`
  ファイル [`src/stim/cmd/command_m2d.cc`](/home/mencotton/qec-project/Stim/src/stim/cmd/command_m2d.cc) / subcommand help `m2d`

### 4-3. 出力データ

- high-level decoder 出力は physical error string ではなく `observable flip` の直接予測。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_decoder_class.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_decoder_class.py) / クラス `CompiledDecoder`
  ファイル [`glue/sample/src/sinter/_predict.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_predict.py) / 関数 `predict_observables`, `predict_observables_bit_packed`

- `predict_observables` の返り値は
  非 bit-packed なら `dtype=np.bool_, shape=(num_shots, num_observables)`
  bit-packed なら `dtype=np.uint8, shape=(num_shots, ceil(num_observables / 8))`
  根拠:
  ファイル [`glue/sample/src/sinter/_predict.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_predict.py) / 関数 `predict_observables`

- `fusion_blossom` / `MWPF` では内部的に edge/hyperedge ごとの `fault_masks` を作るが、これは内部表現でありユーザー向け最終出力ではない。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_fusion_blossom.py) / `fault_masks`
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_mwpf.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_mwpf.py) / `fault_masks`

### 4-4. 運用形態

- 主要運用形態は `logical readout-only decoding`。
  根拠:
  ファイル [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md) / how it works
  ファイル [`glue/sample/src/sinter/_decoding/_stim_then_decode_sampler.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_stim_then_decode_sampler.py) / class `StimThenDecodeSampler`

- もう少し厳密には「detection events から logical observable frame change を予測し、実測 logical observable flip と一致したかで logical error を数える」方式である。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_stim_then_decode_sampler.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_stim_then_decode_sampler.py) / 関数 `classify_discards_and_errors`
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / tutorial text around `count_logical_errors`

- physical data qubit への correction operator を出力する full correction path は確認できない。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_decoding_decoder_class.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding_decoder_class.py) / decoder API
  判定:
  `未実装`

- decoded result をそのまま回路実行へフィードバックして active correction する surface-code pipeline は確認できない。
  根拠:
  repo 内で `sample_decode`, `predict_observables`, `StimThenDecodeSampler` は全て offline 評価フロー
  関数/クラス:
  [`glue/sample/src/sinter/_decoding/_decoding.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding.py) / `sample_decode`
  [`glue/sample/src/sinter/_predict.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_predict.py) / `predict_on_disk`, `predict_observables`
  判定:
  `未実装`

- `Pauli frame update` に近い要素は「observable frame change prediction」にはあるが、prediction の粒度は logical observable 単位であり、full Pauli frame を回路へ戻す API ではない。
  根拠:
  ファイル [`doc/file_format_dem_detector_error_model.md`](/home/mencotton/qec-project/Stim/doc/file_format_dem_detector_error_model.md) / frame change semantics
  ファイル [`glue/sample/src/sinter/_predict.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_predict.py) / `predict_observables`
  判定:
  `logical-level frame prediction` までは明示、`full Pauli frame update` は未確認

### 4-5. パイプライン別まとめ

| pipeline | input data | syndrome definition | output data | operation mode |
| --- | --- | --- | --- | --- |
| `CompiledDetectorSampler.sample(..., separate_observables=True)` | tensor / numpy array | detection events。raw stabilizer 値ではない | detection-event tensor + actual observable-flip tensor | sampling only |
| `sinter.predict_observables` / `sample_decode` + `pymatching` | DEM + detection-event tensor | DEM に対応する detection events | logical observable 直接予測 | logical readout-only |
| `sinter.predict_observables` / `sample_decode` + `fusion_blossom` | DEM + detection-event tensor | graphlike detection events / components | logical observable 直接予測 | logical readout-only |
| `sinter.predict_observables` / `sample_decode` + `mw_parity_factor` / `hypergraph_union_find` | DEM + detection-event tensor | hypergraph を含みうる detection events | logical observable 直接予測 | logical readout-only |
| `stim m2d` / `CompiledMeasurementsToDetectionEventsConverter` | raw measurement data | reference sample との差分で定義される detection events | detection events と任意で observables | pre-decoding conversion |

---

## 5. Neural network 系アルゴリズムの対応

### 5-1. 結論

- neural network 系 decoder の training / inference 実装は repo 内で確認できなかった。
  根拠:
  repo 全体検索で `torch`, `tensorflow`, `keras`, `jax`, `cnn`, `gnn`, `neural decoder` に surface-code decoder 実装として該当するものが見当たらない
  関数/クラス: 該当なし

- よって本項目は
  training: `未対応`
  inference: `未対応`
  合成訓練データ生成: `未対応`
  と整理する。

### 5-2. 補足

- `glue/lattice_surgery` の SAT/SMT 合成器は neural network ではない。
  根拠:
  ファイル [`glue/lattice_surgery/lassynth/sat_synthesis/lattice_surgery_sat.py`](/home/mencotton/qec-project/Stim/glue/lattice_surgery/lassynth/sat_synthesis/lattice_surgery_sat.py) / class `LatticeSurgerySAT`

- `doc/circuit_data_references.md` は外部 repo / データ参照を含むが、この repo の neural decoder 実装根拠にはならない。
  根拠:
  ファイル [`doc/circuit_data_references.md`](/home/mencotton/qec-project/Stim/doc/circuit_data_references.md) / file-level references

---

## 6. ベンチマークの評価内容

### 6-1. benchmark として明示的に確認できたもの

#### 6-1-a. `glue/sample/README.md` Python API example

- 評価対象:
  rotated surface code memory (`surface_code:rotated_memory_x`) の logical error probability vs physical error rate
  根拠:
  ファイル [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md) / function `generate_example_tasks`, `main`

- noise 条件:
  `after_clifford_depolarization=p` のみ
  根拠:
  ファイル [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md) / `generate_example_tasks`

- decoder:
  `pymatching`
  根拠:
  ファイル [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md) / `main`

- 何を failure と数えるか:
  logical observable flip 予測の失敗
  根拠:
  ファイル [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md) / how it works
  ファイル [`glue/sample/src/sinter/_decoding/_stim_then_decode_sampler.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_stim_then_decode_sampler.py) / `classify_discards_and_errors`

- 単位:
  per shot
  根拠:
  ファイル [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md) / plot label `Logical Error Probability (per shot)`

#### 6-1-b. `glue/sample/README.md` Linux CLI example

- 評価対象:
  rotated surface memory (`type=rotated_surface_memory`, basis X) の Monte Carlo logical error statistics
  根拠:
  ファイル [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md) / Linux CLI example

- noise 条件:
  all-four-noise
  `after_clifford_depolarization`
  `after_reset_flip_probability`
  `before_measure_flip_probability`
  `before_round_data_depolarization`
  を全て `p`
  根拠:
  ファイル [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md) / circuit generation snippet

- rounds:
  `rounds=d`
  根拠:
  ファイル [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md) / circuit generation snippet

- bias:
  `b=X` なので logical X memory のみを評価
  根拠:
  ファイル [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md) / filename metadata + `rotated_memory_x`

#### 6-1-c. `doc/getting_started.ipynb` surface-code threshold example

- 評価対象:
  surface code の threshold を circuit noise 下で概算する例
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / lines containing `"under circuit noise"` and threshold narrative

- code:
  `surface_code:rotated_memory_z`
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / `surface_code_tasks` generation cell

- rounds:
  `rounds = d * 3`
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / `surface_code_tasks` generation cell

- distances:
  `[3, 5, 7]`
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / `surface_code_tasks` generation cell

- physical noise sweep:
  `[0.008, 0.009, 0.01, 0.011, 0.012]`
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / `surface_code_tasks` generation cell

- decoder:
  `pymatching`
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / `sinter.collect(..., decoders=['pymatching'])`

- 単位:
  plot 自体は `failure_units_per_shot_func=lambda stat: stat.json_metadata['r']` を使って `per round` logical error rate に再スケールしている。
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / plotting cell
  ファイル [`glue/sample/src/sinter/_plotting.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_plotting.py) / function `plot_error_rate`

- plot title / narrative:
  `"Surface Code Error Rates per Round under Circuit Noise"`
  `"threshold of the surface code is roughly 1%"`
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / plot cell + narrative

#### 6-1-d. `doc/getting_started.ipynb` surface-code footprint example

- 評価対象:
  固定 physical noise `1e-3` で距離依存の logical error を集め、目標 logical error rate に必要な code distance を推定する footprint 例
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / narrative around `"Collect logical error rates from a variety of code distances"`

- code:
  `surface_code:rotated_memory_z`
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / second `surface_code_tasks` generation cell

- rounds:
  `d * 3`
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / second generation cell

- distances:
  `[3, 5, 7, 9]`
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / second generation cell

- collection condition:
  `max_shots=5_000_000`, `max_errors=100`
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / second `sinter.collect` call

### 6-2. benchmark が何を評価しているか

- 主要 benchmark は one-shot readout ではなく `fault-tolerant memory` 実験である。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / tasks `rotated_memory_x`, `rotated_memory_z`, `unrotated_memory_x`, `unrotated_memory_z`
  ファイル [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md) / examples
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / section 9 examples

- logical failure は「decoder の predicted observable flips と sampled observable flips が一致しない shot」の数で定義される。
  根拠:
  ファイル [`glue/sample/src/sinter/_decoding/_stim_then_decode_sampler.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_stim_then_decode_sampler.py) / `classify_discards_and_errors`
  ファイル [`glue/sample/src/sinter/_decoding/_decoding.py`](/home/mencotton/qec-project/Stim/glue/sample/src/sinter/_decoding/_decoding.py) / `_sample_decode_helper_using_memory`, `_streaming_count_mistakes`

### 6-3. 評価前提条件の偏り

- surface-code generator は observable index `0` だけを使う single-logical memory 実験であるため、benchmark も 1 logical qubit / 1 observable basis at a time の評価になる。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / `_finish_surface_code_circuit`

- `rotated_memory_x` と `rotated_memory_z` は別 task なので、logical X と logical Z を同時に一つの benchmark で評価しているわけではない。
  根拠:
  ファイル [`src/stim/gen/gen_surface_code.cc`](/home/mencotton/qec-project/Stim/src/stim/gen/gen_surface_code.cc) / `stim::generate_surface_code_circuit`
  ファイル [`src/stim/cmd/command_gen.cc`](/home/mencotton/qec-project/Stim/src/stim/cmd/command_gen.cc) / CLI help for tasks

- `glue/sample/README.md` の例は X basis 側に偏っている。
  根拠:
  ファイル [`glue/sample/README.md`](/home/mencotton/qec-project/Stim/glue/sample/README.md) / `surface_code:rotated_memory_x`

- `doc/getting_started.ipynb` の threshold / footprint 例は Z basis 側に偏っている。
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / `surface_code:rotated_memory_z`

- threshold 図の前提ノイズモデルは circuit noise であり、phenomenological noise 前提ではない。
  根拠:
  ファイル [`doc/getting_started.ipynb`](/home/mencotton/qec-project/Stim/doc/getting_started.ipynb) / text `"under circuit noise"` + all-four-noise circuit generation cell

- unrotated surface code 用の benchmark script / result 例は `未確認`。
  根拠:
  repo 内の benchmark examples では rotated variant のみを確認
  関数/クラス: 該当なし

- lattice surgery の benchmark result は `未確認`。
  根拠:
  `glue/lattice_surgery` README / demo では synthesizer usage は確認できるが、logical error benchmark CSV/plot までは確認できない
  関数/クラス:
  [`glue/lattice_surgery/README.md`](/home/mencotton/qec-project/Stim/glue/lattice_surgery/README.md) / file-level README
