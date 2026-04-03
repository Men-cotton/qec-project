# Repo Survey: Surface Code Support

更新方針:
- このファイルは段階的に追記する。
- 各節で `明示的記述` / `推論` / `未確認` を分離する。
- 各主張には根拠となるファイルパスと関数・クラス名を付記する。

## 1. Surface code の対応状況と実装範囲

### 1.1 結論

- 対応している。ただし、repo 内で確認できる surface code 対応は単一の専用 surface-code 実装ではなく、主に次の 2 系統に分かれる。
  - `Stim` が生成する rotated surface-code memory circuit を、この repo の `sinter` / overlapping-window decoder で復号する系統。
  - surface code 用 parity-check matrix をテスト入力として、行列ベースの復号器で code-capacity 条件を評価する系統。
  - 根拠:
    - `examples/sinter_example.py` / `generate_example_tasks`: `stim.Circuit.generated(..., code_task="surface_code:rotated_memory_z")`
    - `examples/sinter_example_owd.py` / `generate_example_tasks_surface`: `stim.Circuit.generated(..., code_task="surface_code:rotated_memory_x")`
    - `python_test/test_qcodes.py` / `test_surface_20`: `hx_surface_20.npz`, `lx_surface_20.npz` を読み込み surface code として評価
    - `src_python/ldpc/sinter_decoders/SinterBpOsdDecoder`, `SinterBeliefFindDecoder`, `SinterLsdDecoder`
    - `src_python/ldpc/ckt_noise/PyMatchingOverlappingWindowDecoder`, `BpOsdOverlappingWindowDecoder`, `LsdOverlappingWindowDecoder`

- surface code 関連の主対象ディレクトリ:
  - `examples/`
  - `src_python/ldpc/sinter_decoders/`
  - `src_python/ldpc/ckt_noise/`
  - `python_test/pcms/`
  - `python_test/test_qcodes.py`

### 1.2 対応している code family / variant

#### 明示的記述

- `Stim` 連携例で明示されているのは `surface_code:rotated_memory_z` と `surface_code:rotated_memory_x` である。
  - 根拠:
    - `examples/sinter_example.py` / `generate_example_tasks`
    - `examples/sinter_example_owd.py` / `generate_example_tasks_surface`

#### 推論

- `python_test/pcms/hx_surface_2.npz`, `hx_surface_3.npz`, `hx_surface_4.npz`, `hx_surface_5.npz`, `hx_surface_20.npz` の列数が `d^2 + (d-1)^2` に一致し、対応する `lx_surface_*.npz` が 1 行のみであるため、これらは「単一 logical qubit の rotated planar surface code の片側チェック行列」である可能性が高い。
  - 例:
    - `d=5`: `hx_surface_5.npz` は `(20, 41)`, `lx_surface_5.npz` は `(1, 41)`
    - `41 = 5^2 + 4^2`
    - `d=20`: `hx_surface_20.npz` は `(380, 761)`, `lx_surface_20.npz` は `(1, 761)`
    - `761 = 20^2 + 19^2`
  - 根拠:
    - `python_test/test_qcodes.py` / `test_surface_20`
    - `python_test/pcms/*.npz` の shape

#### 未確認

- repo 内には `XZZX` や `XXZZ` という文字列、またはそれらを切り替える分岐は見当たらない。
- したがって、「XZZX surface code」「XXZZ rotated planar」などの variant を repo が明示対応しているとは確認できない。
  - 根拠:
    - `rg -n "XZZX|XXZZ|rotated|planar" README.md src_python python_test docs/source examples`

### 1.3 実装制約

#### 明示的記述

- `Stim` 連携例は memory experiment 用の rotated surface-code task を使う。
  - 根拠:
    - `examples/sinter_example.py` / `generate_example_tasks`
    - `examples/sinter_example_owd.py` / `generate_example_tasks_surface`

- generic circuit generator `make_css_code_memory_circuit` は arbitrary CSS code 用であり、surface code 専用 API ではない。
  - 根拠:
    - `src_python/ldpc/ckt_noise/css_code_memory_circuit.py` / `make_css_code_memory_circuit`
    - docstring 冒頭: "memory experiment for an arbitrary CSS code"

- `make_css_code_memory_circuit` 自体は multi-logical を受け付ける。`x_logicals`, `z_logicals` は複数行を持て、`test_toric_code_circuit` では `num_observables == 2` を確認している。
  - 根拠:
    - `src_python/ldpc/ckt_noise/css_code_memory_circuit.py` / `make_css_code_memory_circuit`
    - `python_test/test_css_code_memory_circuit.py` / `test_toric_code_circuit`

#### 推論

- surface matrix ベースのテスト入力は単一 logical qubit に限定されている。
  - 根拠:
    - `python_test/pcms/lx_surface_*.npz` はすべて 1 行
    - `python_test/test_qcodes.py` / `test_surface_20`

#### 未確認

- rectangular patch の可否: 未確認
  - 理由:
    - surface 専用の patch generator は repo 内で確認できず、`Stim` 生成 task の詳細制約も repo 内では説明していない。

- odd distance 限定かどうか: 未確認
  - 理由:
    - surface PCM には `d=2,4,20` が含まれる一方、`Stim` 例は `d=5,7,9` のみを使用しており、repo 自身は一貫した距離制約を明示していない。

- open boundary / closed boundary の明示: 未確認
  - 理由:
    - `Stim` 例は `surface_code:rotated_memory_x/z` としか書いておらず、boundary 条件を repo が独自に説明していない。

- lattice surgery support: 未確認ではなく、repo 内には surface-code lattice surgery を扱う API / script / class は見当たらない。
  - 深掘り:
    - `未実装` 寄りの評価である。少なくとも surface-code surgery を対象とした名前・I/O・task 生成・評価コードは見つからない。
    - ただし generic CSS circuit generator を拡張すれば将来表現可能かどうかまでは repo から断定できない。

### 1.4 Capability Matrix

| code family | patch shape | single/multi logical qubit | boundaries | odd distance restriction | repeated syndrome rounds | measurement error support | active correction support | lattice surgery | benchmark scripts | neural decoder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Stim` generated `surface_code:rotated_memory_x/z` | rotated patch (`examples/sinter_example.py` / `generate_example_tasks`, `examples/sinter_example_owd.py` / `generate_example_tasks_surface`) | 未確認 in repo; `memory_x/z` task 名から単一 logical の可能性は高いが repo 内証拠は不足 | 未確認 | 未確認; examples は `d=5,7,9` のみ (`generate_example_tasks`), ただし制約は未記述 | 対応 (`rounds=d` or `(r+1)*d`) | 対応 (`before_measure_flip_probability` を `Stim` に渡す) | いいえ。出力は observables prediction で、回路へ online feedback する API はない (`Sinter*Decoder.decode_via_files`) | 未対応の証拠のみ | あり (`examples/sinter_example.py`, `examples/sinter_example_owd.py`) | なし |
| surface PCM benchmark (`hx_surface_*`, `lx_surface_*`) | 推論: rotated planar single patch | single logical (`lx_surface_*.npz` が 1 行) | 推論: open / planar の可能性が高いが明示なし | 未確認; `d=2,3,4,5,20` の PCM はある | なし | なし | シミュレータ内で物理 correction を評価するが、回路フィードバックではない (`python_test/test_qcodes.py` / `quantum_mc_sim`) | 未対応の証拠のみ | あり (`python_test/test_qcodes.py` / `test_surface_20`) | なし |
| arbitrary CSS memory circuit infrastructure (`make_css_code_memory_circuit`) | surface 専用ではない。任意 CSS | multi-logical 可 (`x_logicals`, `z_logicals`) | user-supplied matrices 依存 | なしと読めるが surface 専用条件ではない | 対応 (`num_rounds`) | 対応 (`before_measure_flip_probability`) | いいえ。生成するのは memory experiment circuit | 未確認。少なくとも専用 API はない | surface 専用 benchmark ではない | なし |

### 1.5 この段階での要約

- repo は surface code の QEC を「専用 surface-code ライブラリ」としてではなく、「generic decoder / generic CSS circuit / Stim integration を surface code に適用する形」で支持している。
- repo 内で明示確認できた surface 系 variant は `Stim` task 名に現れる `surface_code:rotated_memory_x/z` のみであり、`XZZX` や `XXZZ` を repo 自身が区別して実装している証拠はない。
- 行列ベースの surface benchmark は単一 logical qubit の patch を前提としている。

## 2. 対象ノイズモデル

### 2.1 デコーダ実装が仮定するノイズモデル

#### 2.1.1 行列ベース BP / BP+OSD / BP+LSD / BeliefFind / Union-Find

- 明示的記述:
  - `BpDecoderBase` 系は parity-check matrix `pcm` と、各ビットの誤り確率ベクトル `error_channel` または一様 `error_rate` を入力に取る。
  - 入力ベクトル型は surface 系使用箇所では `syndrome` に固定されている。
  - 根拠:
    - `src_python/ldpc/bp_decoder/_bp_decoder.pyx` / `BpDecoderBase.__cinit__`
    - `src_python/ldpc/bposd_decoder/_bposd_decoder.pyx` / `BpOsdDecoder.__cinit__`
    - `src_python/ldpc/bplsd_decoder/_bplsd_decoder.pyx` / `BpLsdDecoder.__cinit__`
    - `src_python/ldpc/belief_find_decoder/_belief_find_decoder.pyx` / `BeliefFindDecoder.__cinit__`

- 解釈:
  - これらの復号器ロジックが直接仮定しているのは「各列に対応する binary fault variable に対する独立チャネル確率」を与えるモデルであり、surface code 専用に `X/Z/Y` を別建てで扱う実装ではない。
  - したがって、matrix-only の surface benchmark では実質的に code-capacity 的な binary data-error model を仮定している。
  - 根拠:
    - `src_python/ldpc/bp_decoder/_bp_decoder.pyx` / `error_channel.setter`, `update_channel_probs`
    - `src_python/ldpc/bposd_decoder/_bposd_decoder.pyx` / `decode`
    - `src_python/ldpc/bplsd_decoder/_bplsd_decoder.pyx` / `decode`
    - `src_python/ldpc/belief_find_decoder/_belief_find_decoder.pyx` / `decode`

- measurement error:
  - matrix-only decoder API 自体には measurement-error 専用変数はない。
  - ただし multi-round parity-check matrix を別途構成すれば measurement fault を「追加列」として埋め込める。
  - 根拠:
    - `src_python/ldpc/monte_carlo_simulation/memory_experiment_v2.py` / `build_multiround_pcm`
    - `src_python/ldpc/bp_decoder/_bp_decoder.pyx` / `BpDecoderBase.__cinit__`

- Y error / correlated error:
  - matrix-only BP/OSD/LSD/BeliefFind は `pcm` の列上の binary variable を復号するため、Y を X/Z に分けて同時復号する surface-code 専用構造は持たない。
  - correlated fault を直接表すには、それを already-expanded な列集合として `pcm` に埋め込む必要がある。
  - 根拠:
    - `src_python/ldpc/bp_decoder/_bp_decoder.pyx` / `Py2BpSparse`, `BpDecoderBase.__cinit__`
    - `src_python/ldpc/bposd_decoder/_bposd_decoder.pyx` / `decode`

- biased noise:
  - decoder core は per-column `error_channel` を受けるため biased noise 自体は表現可能。
  - 根拠:
    - `src_python/ldpc/bp_decoder/_bp_decoder.pyx` / `error_channel.setter`

#### 2.1.2 Stim DEM ベース `sinter` decoders

- 明示的記述:
  - `SinterBpOsdDecoder`, `SinterBeliefFindDecoder`, `SinterLsdDecoder` は `stim.DetectorErrorModel` を `detector_error_model_to_check_matrices` で `check_matrix`, `observables_matrix`, `priors` に変換し、その `check_matrix` を復号する。
  - 根拠:
    - `src_python/ldpc/sinter_decoders/sinter_bposd_decoder.py` / `decode_via_files`
    - `src_python/ldpc/sinter_decoders/sinter_belief_find_decoder.py` / `decode_via_files`
    - `src_python/ldpc/sinter_decoders/sinter_lsd_decoder.py` / `decode_via_files`
    - `src_python/ldpc/ckt_noise/dem_matrices.py` / `detector_error_model_to_check_matrices`

- 解釈:
  - ここで decoder が前提としているのは code-capacity でも phenomenological でもなく、「Stim DEM に列挙された fault mechanism」を列とする detector-hypergraph model である。
  - 各列は detector 集合と observable 集合を持ち、`priors` がその fault の発生確率になる。
  - 根拠:
    - `src_python/ldpc/ckt_noise/dem_matrices.py` / `handle_error`, `DemMatrices`

- measurement error:
  - 対応している。measurement fault は DEM 上の detector event を生む fault mechanism として `check_matrix` に入る。
  - 根拠:
    - `src_python/ldpc/ckt_noise/dem_matrices.py` / `handle_error`
    - `src_python/ldpc/sinter_decoders/sinter_*.py` / `decode_via_files`

- data error のみか:
  - いいえ。DEM に含まれる fault mechanism 全般を扱う。

- correlated error / Y error:
  - `detector_error_model_to_check_matrices` は detector 集合サイズ 2 超の mechanism を hyperedge として `check_matrix` に保持できる。
  - よって BPOSD / LSD / BeliefFind の `sinter` 版は、DEM が表す correlated fault を hyperedge 列として受け取れる。
  - 根拠:
    - `src_python/ldpc/ckt_noise/dem_matrices.py` / `handle_error`, `allow_undecomposed_hyperedges`

- biased noise:
  - 対応している。`priors` は DEM 由来の mechanism ごとの非一様確率である。
  - 根拠:
    - `src_python/ldpc/ckt_noise/dem_matrices.py` / `priors_dict`, `priors`

#### 2.1.3 Overlapping-window + PyMatching

- 明示的記述:
  - `PyMatchingOverlappingWindowDecoder` は `edge_check_matrix` と `edge_observables_matrix` を使う。
  - 重みは `hyperedge_to_edge_matrix @ priors` から `log1p(p) - log(p)` で作る。
  - 根拠:
    - `src_python/ldpc/ckt_noise/pymatching_overlapping_window.py` / `_get_dcm`, `_get_logical_observables_matrix`, `_get_weights`

- 解釈:
  - これは graphlike / matching 型の detector graph を仮定する。
  - measurement error は detector graph の time-like edge として入れられる。
  - correlated hyperedge を直接 MWPM で扱う実装ではない。edge 分解された部分だけを使う。
  - 根拠:
    - `src_python/ldpc/ckt_noise/dem_matrices.py` / `edge_check_matrix`, `hyperedge_to_edge_matrix`
    - `src_python/ldpc/ckt_noise/pymatching_overlapping_window.py` / `_get_dcm`

### 2.2 ベンチマーク / シミュレーションスクリプトが実際に使うノイズモデル

#### 2.2.1 `python_test/test_qcodes.py` の surface benchmark

- 明示的記述:
  - `quantum_mc_sim` は `generate_bsc_error(hx.shape[1], error_rate)` で binary error を生成し、`z = hx @ error % 2` を syndome として使う。
  - 根拠:
    - `python_test/test_qcodes.py` / `quantum_mc_sim`
    - `src_python/ldpc/noise_models/bsc.py` / `generate_bsc_error`

- 結論:
  - 実際に使っているノイズモデルは code-capacity。
  - measurement error: なし
  - data error: あり
  - correlated error: なし
  - Y error: なし
  - biased noise: なし

#### 2.2.2 `examples/sinter_example.py`

- 明示的記述:
  - `stim.Circuit.generated` に
    - `after_clifford_depolarization=p`
    - `after_reset_flip_probability=p`
    - `before_measure_flip_probability=p`
    - `before_round_data_depolarization=p`
    - `code_task="surface_code:rotated_memory_z"`
    - `rounds=d`
    を渡している。
  - 根拠:
    - `examples/sinter_example.py` / `generate_example_tasks`

- 結論:
  - 実際に使っているノイズモデルは circuit-level。
  - measurement error: あり (`before_measure_flip_probability`)
  - reset error: あり (`after_reset_flip_probability`)
  - 1-qubit data depolarization: あり (`before_round_data_depolarization`)
  - 2-qubit Clifford 後 depolarization: あり (`after_clifford_depolarization`)
  - correlated error: ありうる。少なくとも `DEPOLARIZE2` 由来の二体 fault mechanism を含みうる。
  - Y error: depolarizing channel に含まれうる
  - biased noise: なし。全パラメータは同一 `p`

#### 2.2.3 `examples/sinter_example_owd.py` の surface benchmark

- 明示的記述:
  - surface 版タスク生成は `stim.Circuit.generated(..., code_task="surface_code:rotated_memory_x")` を用い、同じく 4 種類の noise parameter をすべて `p` に設定している。
  - 根拠:
    - `examples/sinter_example_owd.py` / `generate_example_tasks_surface`

- 結論:
  - 実際に使っているノイズモデルは circuit-level。
  - measurement error: あり
  - data error: あり
  - correlated error / Y error: `Stim` の depolarizing noise を通じてありうる
  - biased noise: なし

#### 2.2.4 `src_python/ldpc/monte_carlo_simulation/phenomenological_noise_sim.py`

- 明示的記述:
  - `QSS_SimulatorV2` に `per=p`, `ser=p`, `bias=[1.0, 0.0, 0.0]` を渡している。
  - `QSS_SimulatorV2` は data channel と syndrome channel を別々に作り、各 round で noisy syndrome を作る。最終 round は perfect syndrome にしている。
  - 根拠:
    - `src_python/ldpc/monte_carlo_simulation/phenomenological_noise_sim.py`
    - `src_python/ldpc/monte_carlo_simulation/quasi_single_shot_v2.py` / `QSS_SimulatorV2.__init__`, `_single_sample`
    - `src_python/ldpc/monte_carlo_simulation/simulation_utils.py` / `error_channel_setup`, `generate_syndr_err`

- 結論:
  - 実際に使っているノイズモデルは phenomenological。
  - measurement error: あり (`ser`)
  - data error: あり (`per`)
  - correlated error: なし。独立サンプル
  - Y error: `bias` 次第では可能だが、この script の surface/toric 類似設定 `bias=[1,0,0]` では無し
  - biased noise: あり。`bias` ベクトルで制御

### 2.3 この段階での要約

- surface 関連ベンチマークで使われるノイズモデルは 2 本立てである。
  - `python_test/test_qcodes.py`: code-capacity, BSC, perfect syndrome
  - `examples/sinter_example*.py`: circuit-level, repeated rounds, measurement/reset/data/CNOT noise を含む
- repo には phenomenological multi-round simulator もあるが、repo 内で surface 直接例示されているのは toric 系である。

## 3. デコードアルゴリズムの概要

### 3.1 採用アルゴリズムの一覧

- BP
  - 実装:
    - `src_python/ldpc/bp_decoder/_bp_decoder.pyx` / `BpDecoder`
  - surface 関連での役割:
    - 単独利用よりも、BP+OSD / BeliefFind / BP+LSD の前段として使われる。

- BP + OSD
  - 実装:
    - `src_python/ldpc/bposd_decoder/_bposd_decoder.pyx` / `BpOsdDecoder`
    - `src_python/ldpc/sinter_decoders/sinter_bposd_decoder.py` / `SinterBpOsdDecoder`
    - `src_python/ldpc/ckt_noise/bposd_overlapping_window.py` / `BpOsdOverlappingWindowDecoder`

- BeliefFind = BP + Union-Find
  - 実装:
    - `src_python/ldpc/belief_find_decoder/_belief_find_decoder.pyx` / `BeliefFindDecoder`
    - `src_python/ldpc/sinter_decoders/sinter_belief_find_decoder.py` / `SinterBeliefFindDecoder`

- BP + LSD
  - 実装:
    - `src_python/ldpc/bplsd_decoder/_bplsd_decoder.pyx` / `BpLsdDecoder`
    - `src_python/ldpc/sinter_decoders/sinter_lsd_decoder.py` / `SinterLsdDecoder`
    - `src_python/ldpc/ckt_noise/lsd_overlapping_window.py` / `LsdOverlappingWindowDecoder`

- MWPM / PyMatching
  - 実装:
    - `src_python/ldpc/ckt_noise/pymatching_overlapping_window.py` / `PyMatchingOverlappingWindowDecoder`
    - `src_python/ldpc/monte_carlo_simulation/memory_experiment_v2.py` / `get_updated_decoder`

- plain Union-Find
  - 実装:
    - `src_python/ldpc/union_find_decoder/_union_find_decoder.pyx` / `UnionFindDecoder`
  - ただし、repo 内の surface benchmark で直接呼ばれている証拠は見当たらない。surface 関連では BeliefFind の fallback として使われる。

### 3.2 パイプライン別の復号ロジック

#### 3.2.1 Matrix-only surface benchmark (`python_test/test_qcodes.py`)

- パイプライン:
  - binary error `error`
  - syndrome `z = hx @ error % 2`
  - decoder が `z` から物理 correction `decoding` を返す
  - residual `residual = (decoding + error) % 2`
  - logical failure 判定は `lx @ residual % 2`
  - 根拠:
    - `python_test/test_qcodes.py` / `quantum_mc_sim`

- アルゴリズム:
  - `BpOsdDecoder`: BP が収束すれば BP 出力、失敗すれば OSD に fallback
    - 根拠:
      - `src_python/ldpc/bposd_decoder/_bposd_decoder.pyx` / `decode`
  - `BeliefFindDecoder`: BP が収束しなければ Union-Find に fallback
    - `uf_method="peeling"` は各列次数 `<= 2` の point-like syndrome 行列に限定
    - 根拠:
      - `src_python/ldpc/belief_find_decoder/_belief_find_decoder.pyx` / `__cinit__`, `decode`
  - `BpLsdDecoder`: BP が収束しなければ LSD に fallback
    - 根拠:
      - `src_python/ldpc/bplsd_decoder/_bplsd_decoder.pyx` / `decode`

- 重要な性質:
  - ここでは full CSS surface code の X/Z 同時復号はしていない。
  - `hx` と `lx` の 1 側だけを使い、1 種類の binary error chain を復号している。
  - 根拠:
    - `python_test/test_qcodes.py` / `quantum_mc_sim`, `test_surface_20`

#### 3.2.2 `sinter` + DEM + BPOSD / BeliefFind / LSD

- パイプライン:
  - `stim.DetectorErrorModel` を `DemMatrices` に変換
  - `check_matrix` を decoder に渡して correction vector `corr` を得る
  - `observables_matrix @ corr % 2` を observables prediction として返す
  - 根拠:
    - `src_python/ldpc/sinter_decoders/sinter_bposd_decoder.py` / `decode`
    - `src_python/ldpc/sinter_decoders/sinter_belief_find_decoder.py` / `decode`
    - `src_python/ldpc/sinter_decoders/sinter_lsd_decoder.py` / `decode`
    - `src_python/ldpc/ckt_noise/dem_matrices.py` / `detector_error_model_to_check_matrices`

- 物理誤りから syndrome graph への落とし方:
  - repo は `X error`, `Z error`, `Y error` を surface-code 幾何専用に場合分けしていない。
  - 代わりに DEM の各 fault mechanism を
    - detector 集合 `hyperedge_dets`
    - logical observable 集合 `hyperedge_obs`
    にまとめ、1 列の hyperedge として扱う。
  - 根拠:
    - `src_python/ldpc/ckt_noise/dem_matrices.py` / `handle_error`

- X/Z matching の分離有無:
  - 分離していない。
  - 実装は single `check_matrix` / `observables_matrix` による detector-hypergraph 復号であり、surface code の X stabilizer graph と Z stabilizer graph を別々に構成するコードは確認できない。
  - 根拠:
    - `src_python/ldpc/ckt_noise/dem_matrices.py` / `DemMatrices`
    - `src_python/ldpc/sinter_decoders/sinter_*.py` / `decode`

- XXZZ / XZZX など variant の扱い:
  - repo 内には variant ごとの分岐がない。
  - したがって、variant 差は repo が解釈するのではなく、外部の `Stim` circuit / DEM が作る detector connectivity の差としてしか現れない。
  - 根拠:
    - `rg -n "XZZX|XXZZ" README.md src_python python_test docs/source examples`
    - `src_python/ldpc/ckt_noise/dem_matrices.py` / `handle_error`

#### 3.2.3 Overlapping-window decoding (OWD)

- パイプライン:
  - detector データを複数 round にまたがる window に分割
  - 各 window で decoder を走らせる
  - commit 区間の correction を確定
  - その correction を syndrome に反映して次 window へ進む
  - 最終的に correction から observables prediction を計算
  - 根拠:
    - `src_python/ldpc/ckt_noise/base_overlapping_window_decoder.py` / `_corr_multiple_rounds`, `current_round_inds`, `decode`

- MWPM 版:
  - `PyMatchingOverlappingWindowDecoder` は `edge_check_matrix` に対して `Matching.from_check_matrix(...)` を構築
  - 根拠:
    - `src_python/ldpc/ckt_noise/pymatching_overlapping_window.py` / `_init_decoder`

- BP+OSD / BP+LSD 版:
  - `BpOsdOverlappingWindowDecoder` / `LsdOverlappingWindowDecoder` は hyperedge `check_matrix` を window ごとに切り出して復号
  - 根拠:
    - `src_python/ldpc/ckt_noise/bposd_overlapping_window.py` / `_get_dcm`, `_init_decoder`
    - `src_python/ldpc/ckt_noise/lsd_overlapping_window.py` / `_get_dcm`, `_init_decoder`

### 3.3 Y 誤りと両基底 syndrome の扱い

- 明示的記述:
  - `make_css_code_memory_circuit` の docstring は、
    - `include_opposite_basis_detectors=False` でも選んだ logical basis を full distance まで保護するには十分
    - `include_opposite_basis_detectors=True` なら Y errors の情報を利用して性能改善しうる
    と述べている。
  - 根拠:
    - `src_python/ldpc/ckt_noise/css_code_memory_circuit.py` / `make_css_code_memory_circuit` docstring

- 解釈:
  - repo の circuit-level 系は「Y を理論的に別種として特別復号する」のではなく、両基底 detector を含めることで Y fault が残す detection pattern をより豊富に観測する構成を取れる、という設計である。

### 3.4 この段階での要約

- surface 関連で使われる主要アルゴリズムは `BP+OSD`, `BeliefFind`, `BP+LSD`, `MWPM(PyMatching)` である。
- matrix-only surface benchmark は 1 側の check matrix を復号する「独立単一-basis 問題」である。
- `Stim` / DEM 系では physical fault は detector-hyperedge として表現され、repo 内に X/Z の独立 matching graph を surface variant ごとに作る実装は見当たらない。

## 4. 入出力インターフェースとデコードの運用形態

### 4.1 Matrix-only decoders (`BpOsdDecoder`, `BeliefFindDecoder`, `BpLsdDecoder`, `UnionFindDecoder`)

- 入力データ:
  - `np.ndarray` の syndrome ベクトル
  - 長さは parity-check matrix の行数 `m`
  - 根拠:
    - `src_python/ldpc/bposd_decoder/_bposd_decoder.pyx` / `decode`
    - `src_python/ldpc/belief_find_decoder/_belief_find_decoder.pyx` / `decode`
    - `src_python/ldpc/bplsd_decoder/_bplsd_decoder.pyx` / `decode`
    - `src_python/ldpc/union_find_decoder/_union_find_decoder.pyx` / `decode`

- Syndrome の定義:
  - stabilizer 値そのものに相当する binary syndrome
  - `python_test/test_qcodes.py` では `z = hx @ error % 2`
  - difference syndrome / detection event ではない
  - 根拠:
    - `python_test/test_qcodes.py` / `quantum_mc_sim`

- 出力データ:
  - 長さ `n` の physical correction vector
  - 「論理オブザーバブル予測」ではなく、物理ビット列の correction
  - 根拠:
    - `src_python/ldpc/bposd_decoder/_bposd_decoder.pyx` / `decode`
    - `src_python/ldpc/belief_find_decoder/_belief_find_decoder.pyx` / `decode`
    - `src_python/ldpc/bplsd_decoder/_bplsd_decoder.pyx` / `decode`
    - `src_python/ldpc/union_find_decoder/_union_find_decoder.pyx` / `decode`

- 運用形態:
  - full correction に近い。呼び出し元は返ってきた物理 correction を error に足し戻して residual を評価している。
  - ただし回路へ active feedback する API ではない。
  - 根拠:
    - `python_test/test_qcodes.py` / `quantum_mc_sim`

### 4.2 `sinter` decoders (`SinterBpOsdDecoder`, `SinterBeliefFindDecoder`, `SinterLsdDecoder`)

- 入力データ:
  - `stim` / `sinter` の detection event shot data
  - `decode_via_files` では `b8` 形式ファイルから `stim.read_shot_data_file(..., num_detectors=num_dets)` で読む
  - 1 shot あたり shape は `num_dets`
  - 根拠:
    - `src_python/ldpc/sinter_decoders/sinter_bposd_decoder.py` / `decode_via_files`
    - `src_python/ldpc/sinter_decoders/sinter_belief_find_decoder.py` / `decode_via_files`
    - `src_python/ldpc/sinter_decoders/sinter_lsd_decoder.py` / `decode_via_files`

- Syndrome の定義:
  - raw stabilizer 値ではなく detection events
  - `decode_via_files` docstring が "number of detection event bits in each shot" と明記
  - 最終 round の data-qubit readout 再構成は repo 側ではなく、`Stim` circuit / DEM 側の責務
  - 根拠:
    - `src_python/ldpc/sinter_decoders/sinter_*.py` / `decode_via_files`

- 出力データ:
  - observables prediction の binary vector
  - 実装は `observables_matrix @ corr % 2`
  - 物理エラー列そのものは `sinter` API 外へ返さない
  - 根拠:
    - `src_python/ldpc/sinter_decoders/sinter_bposd_decoder.py` / `decode`
    - `src_python/ldpc/sinter_decoders/sinter_belief_find_decoder.py` / `decode`
    - `src_python/ldpc/sinter_decoders/sinter_lsd_decoder.py` / `decode`

- 運用形態:
  - logical readout-only decoding
  - output は observables flip の予測であり、Pauli frame update や active correction command を返す形ではない
  - 用途は fault-tolerant memory / logical observable 評価

### 4.3 Overlapping-window decoders

- 入力データ:
  - detector event bit 列
  - `decode` は `np.ndarray` 1 shot, `decode_batch` は `(num_shots, num_detectors)` または bit-packed
  - 根拠:
    - `src_python/ldpc/ckt_noise/base_overlapping_window_decoder.py` / `decode`, `decode_batch`
    - `src_python/ldpc/ckt_noise/sinter_overlapping_window_decoder.py` / `decode_shots_bit_packed`, `decode_via_files`

- Syndrome の定義:
  - detection events
  - `BaseOverlappingWindowDecoder.decode` docstring が「sampled detector data」を想定
  - 根拠:
    - `src_python/ldpc/ckt_noise/base_overlapping_window_decoder.py` / `decode`

- 出力データ:
  - public API 出力は observables prediction
  - 内部的には edge/hyperedge correction vector `corr` を計算してから `logical_observables_matrix @ corr % 2`
  - 根拠:
    - `src_python/ldpc/ckt_noise/base_overlapping_window_decoder.py` / `decode`, `_corr_multiple_rounds`

- 運用形態:
  - window ごとに partial correction を commit する simulated streaming decode
  - ただし circuit へ online feedback する I/O はない。最終的に出すのは observables prediction
  - 根拠:
    - `src_python/ldpc/ckt_noise/base_overlapping_window_decoder.py` / `_corr_multiple_rounds`

### 4.4 `make_css_code_memory_circuit` が定義する syndrome / observable

- 入力データ:
  - `x_stabilizers`, `z_stabilizers`, `x_logicals`, `z_logicals` は sparse matrix
  - 根拠:
    - `src_python/ldpc/ckt_noise/css_code_memory_circuit.py` / `make_css_code_memory_circuit`

- Syndrome の定義:
  - bulk detector は前 round と今 round の measurement の差分
  - `body` で `DETECTOR` targets に 2 回分の `rec` を渡している
  - 根拠:
    - `src_python/ldpc/ckt_noise/css_code_memory_circuit.py` / `make_css_code_memory_circuit`

- 最終 round 再構成:
  - はい。final detector は最後の ancilla 測定結果に加えて、対応する stabilizer support 上の data-qubit readout を束ねて作る
  - 根拠:
    - `src_python/ldpc/ckt_noise/css_code_memory_circuit.py` / `tail` 構築部

- 出力データ:
  - circuit 自体は `OBSERVABLE_INCLUDE` により logical observable を定義する
  - 根拠:
    - `src_python/ldpc/ckt_noise/css_code_memory_circuit.py` / `tail` 構築部

- 運用形態:
  - fault-tolerant memory experiment circuit の生成
  - active correction ループは含まない

### 4.5 `QSS_SimulatorV2` / `decode_multiround`

- 入力データ:
  - raw syndrome matrix `syndrome_mat` with shape `(num_checks, repetitions)`
  - 各列は 1 round の stabilizer measurement result
  - 根拠:
    - `src_python/ldpc/monte_carlo_simulation/quasi_single_shot_v2.py` / `_single_sample`

- Syndrome の定義:
  - decoder に渡す直前に difference syndrome へ変換している
  - `diff_syndrome[:, 1:] = (syndrome[:, 1:] - syndrome[:, :-1]) % 2`
  - 根拠:
    - `src_python/ldpc/monte_carlo_simulation/memory_experiment_v2.py` / `decode_multiround`

- 最終 round:
  - 最後の round は perfect syndrome にしている
  - これは final data readout 再構成ではなく、最後の stabilizer measurement を noiseless にする設計
  - 根拠:
    - `src_python/ldpc/monte_carlo_simulation/quasi_single_shot_v2.py` / `_single_sample`
    - `src_python/ldpc/monte_carlo_simulation/memory_experiment_v2.py` / `decode_multiround`

- 出力データ:
  - `decode_multiround` は physical correction vector `decoded` を返す
  - last-round でない場合は commit region の correction のみ返す
  - 根拠:
    - `src_python/ldpc/monte_carlo_simulation/memory_experiment_v2.py` / `decode_multiround`

- 運用形態:
  - full correction in simulation
  - correction を `err = (err + corr) % 2` で内部状態へ反映する
  - ただし外部量子回路へ active feedback する実装ではない
  - 根拠:
    - `src_python/ldpc/monte_carlo_simulation/quasi_single_shot_v2.py` / `_single_sample`

### 4.6 この段階での要約

- repo の surface 関連パイプラインは大きく 2 つに分かれる。
  - matrix-only: raw syndrome in, physical correction out
  - circuit/DEM/sinter: detection events in, logical observables prediction out
- `make_css_code_memory_circuit` は final detector を data-qubit readout から再構成する。
- `QSS_SimulatorV2` は raw repeated syndrome を差分化してから multi-round decoder に渡す。

## 5. Neural network 系アルゴリズムの対応

### 5.1 結論

- 該当なし。
- repo 内に CNN / GNN / Transformer 等の neural decoder 実装は確認できない。
- training と inference の片対応実装も確認できない。
- 合成訓練データ生成コードも確認できない。
- 根拠:
  - `rg -n "torch|tensorflow|keras|jax|gnn|graph neural|cnn|neural|pytorch|inference|train|dataset" src_python python_test examples docs README.md`

### 5.2 未対応理由

- `未実装` と判断するのが妥当。
- 理由:
  - decoder 実装として見つかるのは `BpOsdDecoder`, `BeliefFindDecoder`, `BpLsdDecoder`, `UnionFindDecoder`, `PyMatchingOverlappingWindowDecoder` など classical / combinatorial 手法のみであり、model 定義・学習 loop・optimizer・dataset pipeline が存在しない。

## 6. ベンチマークの評価内容

### 6.1 repo 内に benchmark 結果ファイルはあるか

- 少なくとも調査時点の tree には surface benchmark の結果 CSV や `results/` ディレクトリは存在しない。
- 見つかった CSV は `cpp_test/test_inputs/*.csv` のみで、benchmark 出力ではない。
- 根拠:
  - `find . -type f \\( -name '*.csv' -o -path './results/*' \\) | sort`

### 6.2 `python_test/test_qcodes.py` / `test_surface_20`

- 何を評価しているか:
  - logical memory に相当する code-capacity 論理失敗率
  - 1 回の perfect syndrome から復号し、`lx @ residual` で logical failure を判定
  - 根拠:
    - `python_test/test_qcodes.py` / `quantum_mc_sim`, `test_surface_20`

- 評価対象の偏り:
  - `hx` と `lx` のみを使うため、full CSS の X/Z 両側を同時には評価していない
  - 1 側の error class / logical operator class に偏った評価
  - 根拠:
    - `python_test/test_qcodes.py` / `quantum_mc_sim`, `test_surface_20`

- 前提条件:
  - rounds: 1 相当
  - measurement error: なし
  - 誤りモデル: BSC / code-capacity
  - surface 入力: print 文では `[[761, 1, 20]] Surface`
  - decoders: `BpOsdDecoder`, `BeliefFindDecoder`, `BpLsdDecoder`
  - `error_rate = 0.05`, `run_count = 1000`, `max_iter = 5`
  - 根拠:
    - `python_test/test_qcodes.py` / `test_surface_20`

### 6.3 `examples/sinter_example.py`

- 何を評価しているか:
  - rotated surface-code memory 実験の logical error rate
  - `sinter.plot_error_rate` により physical error rate `p` に対する logical error rate を描画する
  - 根拠:
    - `examples/sinter_example.py` / `generate_example_tasks`, `main`

- 前提条件:
  - task: `code_task="surface_code:rotated_memory_z"`
  - basis 偏り: Z-memory のみ
  - rounds: `d`
  - distances: `5, 7, 9`
  - `p`: `0.001` から `0.009` まで step `0.002`
  - `max_shots = 20_000`, `max_errors = 100`
  - 誤りモデル: circuit-level
  - 根拠:
    - `examples/sinter_example.py` / `generate_example_tasks`, `main`

- 補足:
  - script は custom decoder key として `"bplsd"` を登録している一方、3 枚目 plot の filter は `stat.decoder == "lsd"` である。
  - したがって、repo 内コードだけからは 3 枚目の LSD 系列が正しく描画されるか未確認。
  - 根拠:
    - `examples/sinter_example.py` / `main`

### 6.4 `examples/sinter_example_owd.py`

- 何を評価しているか:
  - rotated surface-code memory 実験の logical error rate
  - ただし surface 版 task generator は `decoder=f"pymatching_owd_d{d}_r{r}"` を固定しており、surface で実際に評価しているのは PyMatching OWD のみ
  - 根拠:
    - `examples/sinter_example_owd.py` / `generate_example_tasks_surface`

- 前提条件:
  - task: `code_task="surface_code:rotated_memory_x"`
  - basis 偏り: X-memory のみ
  - rounds: `(r + 1) * d`
  - OWD window: `2 * d`
  - OWD commit: `d`
  - `num_checks = d^2 - 1` を PyMatching OWD に渡す
  - decodings parameter `r`: `2, 3`
  - `ds = [12]`
  - `ps = geomspace(2e-3, 0.011, 9)`
  - 誤りモデル: circuit-level
  - 根拠:
    - `examples/sinter_example_owd.py` / `generate_decoders`, `generate_example_tasks_surface`, `main`

- 評価対象の偏り:
  - X-memory のみ
  - 単一距離 `d=12` のみ
  - OWD の `decodings` 数比較に偏る

### 6.5 `phenomenological_noise_sim.py`

- 何を評価しているか:
  - repo 内では toric code の phenomenological logical memory 実験
  - surface code benchmark ではない
  - 根拠:
    - `src_python/ldpc/monte_carlo_simulation/phenomenological_noise_sim.py`

- 前提条件:
  - distances: `3, 5, 7`
  - `codename = "2DTC"`
  - `per = ser = p`
  - `bias = [1.0, 0.0, 0.0]`
  - `rounds = (decoding_rds + 1) * dist`
  - `repetitions = 2 * dist`
  - last round perfect syndrome
  - 根拠:
    - `src_python/ldpc/monte_carlo_simulation/phenomenological_noise_sim.py`
    - `src_python/ldpc/monte_carlo_simulation/quasi_single_shot_v2.py` / `_single_sample`

### 6.6 この段階での要約

- repo には surface benchmark の「結果ファイル」は同梱されていないが、benchmark script は存在する。
- surface benchmark は 2 系統に分かれる。
  - code-capacity / perfect syndrome / single-basis matrix benchmark
  - circuit-level / repeated rounds / memory experiment benchmark (`Stim` / `sinter`)
- circuit-level surface benchmark も basis ごとの memory task に偏っており、multi-logical interaction や lattice surgery の評価は含まれない。
