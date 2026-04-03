# Repo Investigation: Surface Code Support and Decoding Pipelines

## 1. Surface code の対応状況と実装範囲

### 1.1 対応有無

- 結論: この repo は surface code の QEC に対応している。
  - 明示的記述:
    - `src/rotated_surface_model.py` に `RotSurCode` クラスがあり、単一のコード状態 `qubit_matrix` と syndrome 配列 `plaquette_defects` を保持する。[根拠: `src/rotated_surface_model.py`, `RotSurCode.__init__`]
    - `generate_data.py` は `params['code'] == 'rotated'` のとき `RotSurCode(params['size'])` を生成してデータを作る。[根拠: `generate_data.py`, `generate`]
    - `2drnn.py` / `dilatedrnn.py` はともに `RotSurCode` を import して評価対象のコードとして使う。[根拠: `2drnn.py`, module scope; `dilatedrnn.py`, module scope]
  - 対象ディレクトリ:
    - `src/` (`src/rotated_surface_model.py`, `src/mcmc.py`, `src/mcmc_alpha.py`)
    - ルート直下のパイプライン (`generate_data.py`, `decoders.py`, `2drnn.py`, `dilatedrnn.py`, `plot.py`, `run.py`)

### 1.2 対応する code family / variant

- 結論: 現在の実装本体として確認できるのは、square rotated planar surface code の CSS 型バリアント 1 種のみ。
  - 明示的記述:
    - `RotSurCode` は plaquette の演算子種別を `op = 1` または `op = 3` で切り替える。[根拠: `src/rotated_surface_model.py`, `_find_syndrome`, `_apply_stabilizer`]
    - full plaquette では `(row, col)` の parity に応じて `X` と `Z` が交互になる。[根拠: `src/rotated_surface_model.py`, `_find_syndrome` lines 177-188; `_apply_stabilizer` lines 276-287]
    - 境界では 4 辺に half stabilizer を配置している。[根拠: `src/rotated_surface_model.py`, `RotSurCode.syndrome` lines 69-85; `_find_syndrome` lines 189-201]
  - 推論:
    - 交互の `X`/`Z` plaquette を持つ rotated CSS patch であり、XZZX のような非 CSS ねじれ配置ではない。`XZZX` を構成する別クラスや別 syndrome 規則は repo 内で確認できない。[根拠: `src/rotated_surface_model.py`, `_find_syndrome`; `src/mcmc.py`, `Chain.update_chain_fast`]

### 1.3 実装制約

- 単一 logical qubit:
  - 明示的記述:
    - `RotSurCode.nbr_eq_classes = 4` で、等価類は 4 通りに固定される。[根拠: `src/rotated_surface_model.py`, `RotSurCode`]
    - `_define_equivalence_class` は first row と first column の parity だけで logical class を決める。[根拠: `src/rotated_surface_model.py`, `_define_equivalence_class`]
  - 推論:
    - 2 logical qubit 以上を持つ multi-patch / merged patch は対象外。単一パッチの 4 等価類だけを扱う設計である。[根拠: `src/rotated_surface_model.py`, `_define_equivalence_class`; `generate_data.py`, `generate`]

- patch shape:
  - 明示的記述:
    - `qubit_matrix` は `(size, size)` の正方行列で初期化される。[根拠: `src/rotated_surface_model.py`, `RotSurCode.__init__`]
  - 推論:
    - rectangular patch は未実装。引数が単一の `size` のみで、長方形を表す独立な縦横パラメータが存在しない。[根拠: `src/rotated_surface_model.py`, `RotSurCode.__init__`; `generate_data.py`, `generate`]

- boundaries:
  - 明示的記述:
    - 4 辺の half stabilizer を個別に計算しており、周期境界のコードは使っていない。[根拠: `src/rotated_surface_model.py`, `RotSurCode.syndrome`; `_find_syndrome`]
  - 推論:
    - open boundary rotated planar patch である。toric / periodic boundary の実装本体は repo 内で確認できない。[根拠: `src/rotated_surface_model.py`, `RotSurCode.syndrome`; `src/mcmc.py`, `Chain.update_chain_fast`]

- odd distance restriction:
  - 明示的記述:
    - boundary half stabilizer の個数は `int((size - 1)/2)` に依存する。[根拠: `src/rotated_surface_model.py`, `RotSurCode.syndrome`; `_apply_random_stabilizer`; `_apply_stabilizers_uniform`]
    - 実行例や既存データは `size = 3, 5` を使う。[根拠: `run.py`, module `__main__`; `2drnn.py`, module `__main__`; `dilatedrnn.py`, module `__main__`]
  - 推論:
    - odd distance を暗黙に前提にしている可能性が高い。even `size` を禁止する assert は無いが、`int((size - 1)/2)` により even size では境界要素数が自然に一致しないため、仕様としては odd distance 想定と読むのが妥当。[根拠: `src/rotated_surface_model.py`, `RotSurCode.syndrome`, `_apply_random_stabilizer`, `_apply_stabilizers_uniform`]

- repeated rounds / measurement errors:
  - 明示的記述:
    - syndrome は単一の 2 次元 `plaquette_defects` 配列で保持され、時間方向 index が無い。[根拠: `src/rotated_surface_model.py`, `RotSurCode.__init__`, `RotSurCode.syndrome`]
  - 推論:
    - repeated syndrome rounds を持つ 3D matching graph や measurement error モデルは対象外。[根拠: `src/rotated_surface_model.py`, `RotSurCode.syndrome`; `generate_data.py`, `generate`]

### 1.4 Capability Matrix

| code family | patch shape | single/multi logical qubit | boundaries | odd distance restriction | repeated syndrome rounds | measurement error support | active correction support | lattice surgery | benchmark scripts | neural decoder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rotated planar CSS patch (XXZZ-like alternating X/Z plaquettes) | square only | single logical qubit only | open boundaries | 暗黙に odd 想定 | no | no | no | no | yes | yes |
| XZZX rotated patch | 未確認 | 未確認 | 未確認 | 未確認 | 未確認 | 未確認 | 未確認 | 未確認 | 既存結果ファイル名上は存在 | 実装本体は未確認 |

### 1.5 Matrix 各列の判定根拠

- rotated planar CSS patch row:
  - `benchmark scripts = yes`
    - 明示的記述: `run.py` が `params = {'code': 'rotated', 'method': 'EWD', ...}` で `generate(...)` を実行する。[根拠: `run.py`, module `__main__`]
    - 明示的記述: `plot.py` は `data/EWD_reference_size3*`, `data/Rnn_2dtest_d3*`, `data/Rnn_test_entill2*` を読み、`define_eq_rot` と `argmax(eq_distr[:4])` を比較して失敗率を描画する。[根拠: `plot.py`, module scope]
  - `neural decoder = yes`
    - 明示的記述: `2drnn.py` と `dilatedrnn.py` に TensorFlow ベースの RNN wavefunction 実装があり、各サンプルから `df_eq_distr` を作る。[根拠: `2drnn.py`, `MDRNNWavefunction`, module `__main__`; `dilatedrnn.py`, `DilatedRNNWavefunction`, module `__main__`]
  - `active correction support = no`
    - 明示的記述: 既存パイプラインの出力は等価類分布 `df_eq_distr` であり、補正演算列を回路へ返す経路は確認できない。[根拠: `generate_data.py`, `generate`; `2drnn.py`, module `__main__`; `dilatedrnn.py`, module `__main__`]
    - 推論: 用途は logical class 推定による logical readout / memory 評価であり、回路フィードバック型 active correction ではない。
  - `lattice surgery = no`
    - 明示的記述: 単一 `qubit_matrix` と 4 等価類のみを扱い、複数パッチ相互作用 API が無い。[根拠: `src/rotated_surface_model.py`, `RotSurCode`, `_define_equivalence_class`]
    - 推論: 単に未実装というより、現設計は単一パッチ error-chain 分布評価に閉じており対象外。

- XZZX row:
  - 明示的記述:
    - `src/mcmc.py` には `xzzx_code` 分岐があるが、当該クラス定義や `_update_chain_fast_xzzx` は repo 内で見つからない。[根拠: `src/mcmc.py`, `Chain.update_chain_fast`]
    - `data/EWD_xzzx3_alpha1_*.xz` という既存結果ファイル群は存在する。[根拠: `data/` 配下ファイル名]
  - 推論:
    - 以前の別実装・別 repo 由来の成果物が混在している可能性はあるが、この repo 単体から再現可能な XZZX 実装本体は未確認。

## 2. 対象ノイズモデル

### 2.1 総括

- 結論:
  - 現行の end-to-end 実装が前提とするのは code-capacity 系モデルのみである。
  - repeated syndrome rounds, measurement error, circuit-level fault propagation を扱う実装は確認できない。
  - 実際の benchmark では、名前上 `alpha` とされていても pure X noise に退化している経路が複数ある。

### 2.2 デコーダ実装が仮定するノイズモデル

#### A. EWD decoder (`decoders.py` + `src/mcmc_alpha.py`)

- 前提モデル:
  - 結論: code-capacity の `alpha` モデルを仮定する。
  - 明示的記述:
    - `EWD_alpha` は引数として `pz_tilde` と `alpha` を取り、`beta = -log(pz_tilde)` と `eff_len = n_z + alpha * (n_x + n_y)` を使って各等価類へ重み付けする。[根拠: `decoders.py`, `EWD_alpha`, `EWD_droplet_alpha`]
    - `Chain_alpha` の受理確率は `p = pz_tilde**(dz + alpha*(dx + dy))` で計算される。[根拠: `src/mcmc_alpha.py`, `Chain_alpha.update_chain_fast`, `_update_chain_fast_rotated`]
  - 推論:
    - 物理エラーは独立な data-qubit Pauli error の effective-length モデルとして扱われ、circuit fault や time-like detection event graph は使っていない。

- measurement error:
  - 明示的記述:
    - 更新対象は `qubit_matrix` のみで、syndrome 測定値そのものに独立誤りを入れるロジックが無い。[根拠: `src/rotated_surface_model.py`, `RotSurCode.generate_random_error`, `RotSurCode.syndrome`; `src/mcmc_alpha.py`, `_update_chain_fast_rotated`]
  - 結論: 非対応。

- data error / Y / biased noise:
  - 明示的記述:
    - `chain_lengths()` は `n_x, n_y, n_z` を別々に数え、`EWD_droplet_alpha` は `n_y` を `n_x` と同じ係数 `alpha` で扱う。[根拠: `src/rotated_surface_model.py`, `RotSurCode.chain_lengths`; `decoders.py`, `EWD_droplet_alpha`]
  - 推論:
    - Y error は「X 側と同じコスト」でしか扱われず、独立の Y 特化モデルではない。
    - biased noise 専用 EWD は現行の top-level decoder としては未接続。`src/mcmc_biased.py` は存在するが `planar_model` import に依存しており、この repo 単体では end-to-end 動作は未確認。[根拠: `src/mcmc_biased.py`, module scope]

- correlated error:
  - 明示的記述:
    - 受理確率は `dx, dy, dz` の個数だけで決まり、ペア相関やフックエラーの相関構造を表す項が無い。[根拠: `src/mcmc_alpha.py`, `_update_chain_fast_rotated`; `decoders.py`, `EWD_alpha`]
  - 結論: 相関誤り非対応。

#### B. RNN / VNA 系 (`2drnn.py`, `dilatedrnn.py`)

- 前提モデル:
  - 結論: code-capacity の binary site-error モデルを仮定する。
  - 明示的記述:
    - 両スクリプトとも RNN 出力層は各 site で 2 値 softmax であり、`inputdim=2` として binary sample を生成する。[根拠: `2drnn.py`, `MDRNNWavefunction.__init__`, `MDRNNWavefunction.sample`; `dilatedrnn.py`, `DilatedRNNWavefunction.__init__`, `DilatedRNNWavefunction.sample`]
    - local energy は `syndrome mismatch penalty + np.sum(samples)` で定義され、Pauli 種別ごとの 4 値状態は扱わない。[根拠: `2drnn.py`, `return_local_energies`; `dilatedrnn.py`, `return_local_energies`]
  - 推論:
    - これらの neural decoder は一般の Pauli X/Y/Z 誤り分布ではなく、binary occupancy としての単一誤り種別しか直接表現できない。

- syndrome conditioning:
  - 明示的記述:
    - RNN cell は `init_code.plaquette_defects` の近傍 4 要素を定数入力として使う。[根拠: `2drnn.py`, `MDTensorizedRNNCell.call`; `dilatedrnn.py`, `CustomRNNCell.call`]
  - 推論:
    - 入力は detection event ではなく単一ラウンドの stabilizer defect 配列である。

- measurement error / correlated error / Y:
  - 明示的記述:
    - sample は binary tensor で、`update_matrix(sample)` にそのまま渡される。[根拠: `2drnn.py`, module `__main__` lines 773-780; `dilatedrnn.py`, module `__main__` lines 412-419]
  - 結論:
    - measurement error: 非対応
    - correlated error: 非対応
    - Y error: 明示的には非対応

### 2.3 ベンチマーク / シミュレーションスクリプトが実際に使うノイズモデル

#### A. `generate_data.py` + `run.py` による EWD ベンチマーク

- 前提モデル:
  - 結論: code-capacity, single-shot data-qubit error のみ。
  - 明示的記述:
    - `generate()` は `RotSurCode.generate_random_error(p_x, p_y, p_z)` を 1 回呼んで初期誤りを作る。[根拠: `generate_data.py`, `generate`]
    - `run.py` は `params['noise'] = 'alpha'` として `generate(...)` を呼ぶ。[根拠: `run.py`, module `__main__`]

- 実際の誤り注入:
  - 明示的記述:
    - `get_individual_error_rates()` の `alpha` 分岐では一旦 `p_z`, `p_x`, `p_y` を計算した後、`p_y = p_z = 0` に上書きしている。[根拠: `generate_data.py`, `get_individual_error_rates`]
    - `generate()` の EWD 分岐は `noise == 'alpha'` 以外を拒否する。[根拠: `generate_data.py`, `generate`]
  - 結論:
    - 既存 EWD benchmark スクリプトは、ラベル上は `alpha` noise だが、実際の注入ノイズは pure X noise。

- depolarizing / biased / Y:
  - 明示的記述:
    - `get_individual_error_rates()` 自体は `depolarizing`, `alpha`, `biased` を受け付ける。[根拠: `generate_data.py`, `get_individual_error_rates`]
    - ただし EWD 生成経路は `alpha` 以外で `ValueError` を投げる。[根拠: `generate_data.py`, `generate`]
  - 結論:
    - generate 側の補助関数には depolarizing / biased の式があるが、現行 EWD benchmark では使われない。

#### B. `2drnn.py` の benchmark loop

- 前提モデル:
  - 明示的記述:
    - `params['noise'] = 'alpha'` とメタデータに書いている。[根拠: `2drnn.py`, module `__main__`]
    - 実際の注入は `p_x = ps[p_idx]`, `p_y = 0`, `p_z = 0` のまま `RotSurCode.generate_random_error(...)` を呼ぶ。[根拠: `2drnn.py`, module `__main__` lines 637-641, 697-716]
  - 結論:
    - 実際のノイズは pure X code-capacity noise。

- measurement error / correlated error:
  - 明示的記述:
    - 時間方向 index や測定誤りサンプルは無い。[根拠: `2drnn.py`, `return_local_energies`, module `__main__`; `src/rotated_surface_model.py`, `RotSurCode.syndrome`]
  - 結論: 非対応。

#### C. `dilatedrnn.py` の benchmark loop

- 前提モデル:
  - 明示的記述:
    - `params['noise'] = 'alpha'` とメタデータに書いている。[根拠: `dilatedrnn.py`, module `__main__`]
    - 実際の注入は `p_x = ps[p_idx]`, `p_y = 0`, `p_z = 0` のまま `RotSurCode.generate_random_error(...)` を呼ぶ。[根拠: `dilatedrnn.py`, module `__main__` lines 277-281, 337-358]
  - 結論:
    - 実際のノイズは pure X code-capacity noise。

### 2.4 未確認事項

- `biased` noise の end-to-end benchmark:
  - 未確認。
  - 理由:
    - `generate_data.py` には biased rate 変換があるが、EWD 分岐は `alpha` のみを許可する。
    - `src/mcmc_biased.py` はあるが、依存する `.planar_model` が repo 内に見当たらないため、本 repo 単体で到達可能な評価経路か確認できない。

## 3. デコードアルゴリズムの概要

### 3.1 実際に確認できるパイプライン

- EWD alpha decoder
  - 明示的記述:
    - `generate_data.py` の `params['method'] == "EWD"` 分岐は `EWD_alpha(...)` を呼ぶ。[根拠: `generate_data.py`, `generate`]
    - `EWD_alpha` は各 logical equivalence class ごとに `Chain_alpha` を初期化し、`to_class(eq)` で同一 syndrome の別等価類へ移した上で重みを積算する。[根拠: `decoders.py`, `EWD_alpha`; `src/rotated_surface_model.py`, `RotSurCode.to_class`]
  - 推論:
    - これは MWPM ではなく、stabilizer move による error chain サンプリングと等価類ごとの重み比較を使う decoder である。

- 2D RNN VNA decoder
  - 明示的記述:
    - `2drnn.py` は `MDRNNWavefunction` と `MDTensorizedRNNCell` を定義し、`optstep` による反復最適化で sample 分布を更新する。[根拠: `2drnn.py`, `MDTensorizedRNNCell`, `MDRNNWavefunction`, module `__main__`]
  - 推論:
    - これは supervised classifier ではなく、各 syndrome ごとに変分分布を最適化する VNA 型 decoder である。

- 1D dilated RNN VNA decoder
  - 明示的記述:
    - `dilatedrnn.py` は `DilatedRNNWavefunction` と `CustomRNNCell` を定義し、同様に `optstep` を回している。[根拠: `dilatedrnn.py`, `CustomRNNCell`, `DilatedRNNWavefunction`, module `__main__`]
  - 推論:
    - 2D tensorized RNN の代わりに、flatten した binary chain へ dilated recurrence をかける別実装である。

- MWPM / CNN / GNN
  - 明示的記述:
    - `generate_data.py` に `mwpm_init` と `class_sorted_mwpm` 参照はあるが、実装本体は repo 内で確認できない。[根拠: `generate_data.py`, `generate`]
  - 未確認:
    - `CNN`, `GNN`, `PyMatching` 等の decoder 実装入口は、今回確認した `README.md`, ルート直下スクリプト群, `src/` 配下では見つからなかった。[根拠: `README.md`, module scope; ルート直下ファイル一覧; `src/` 配下ファイル一覧]
  - 結論:
    - 現 repo で実体が確認できる decoder は EWD と VNA-RNN 系のみ。

### 3.2 syndrome への落とし方

- 明示的記述:
  - `_find_syndrome()` は plaquette 演算子 `op` と各 qubit 上の Pauli 値を比較し、`old_qubit != 0 and old_qubit != op` のとき defect bit を反転する。[根拠: `src/rotated_surface_model.py`, `_find_syndrome`]
  - full plaquette の `op` は parity によって `1` と `3` が交互に並ぶ。[根拠: `src/rotated_surface_model.py`, `_find_syndrome`]
- 推論:
  - `op = 1` plaquette は X stabilizer なので Z/Y error に反応し、`op = 3` plaquette は Z stabilizer なので X/Y error に反応する。
  - 現実装は CSS 型の X-check / Z-check defect を 1 つの 2D defect array 内に交互配置している。

### 3.3 EWD alpha decoder の流れ

1. `RotSurCode` で与えられた error chain から syndrome を固定する。
2. `to_class(eq)` で同じ syndrome を持つ 4 つの等価類の代表へ移す。
3. 各等価類で `Chain_alpha.update_chain_fast()` を回し、stabilizer move による chain を探索する。
4. 各 chain の `eff_len = n_z + alpha (n_x + n_y)` を計算する。
5. `exp(-beta * eff_len)` を足し上げ、4 等価類の分布を返す。

- 根拠:
  - `decoders.py`, `EWD_alpha`, `EWD_droplet_alpha`
  - `src/mcmc_alpha.py`, `Chain_alpha.update_chain_fast`, `_update_chain_fast_rotated`
  - `src/rotated_surface_model.py`, `RotSurCode.to_class`, `RotSurCode.chain_lengths`

### 3.4 RNN VNA decoder の流れ

#### A. 2D RNN

1. `init_code.plaquette_defects` を cell 内の条件付け入力として使う。
2. `MDRNNWavefunction.sample()` が binary `samples[num_samples, Nx, Ny]` を生成する。
3. `return_local_energies()` が syndrome mismatch と site-count から局所エネルギーを作る。
4. `Floc = Eloc + T * log_probs` を用いた変分目的で `optstep` を反復する。
5. 最終 sample 群を `define_equivalence_class()` に通し、4 等価類のヒストグラム `df_eq_distr` を出す。

- 根拠:
  - `2drnn.py`, `MDTensorizedRNNCell.call`, `MDRNNWavefunction.sample`, `return_local_energies`, module `__main__`

#### B. 1D dilated RNN

1. syndrome 条件付けは `CustomRNNCell.call()` 内で `init_code.plaquette_defects` の近傍 4 値を使う。
2. `DilatedRNNWavefunction.sample()` が flattened binary chain `samples[num_samples, N]` を生成する。
3. `return_local_energies()` で reshape 後の syndrome mismatch と site-count を使い局所エネルギーを作る。
4. 同じ変分目的で `optstep` を反復する。
5. sample を `N_new x N_new` に戻して 4 等価類の分布 `df_eq_distr` を集計する。

- 根拠:
  - `dilatedrnn.py`, `CustomRNNCell.call`, `DilatedRNNWavefunction.sample`, `return_local_energies`, module `__main__`

### 3.5 XXZZ / XZZX バリアントの扱い

- rotated CSS / XXZZ-like variant:
  - 明示的記述:
    - full plaquette が parity で `X` / `Z` に交互化される。[根拠: `src/rotated_surface_model.py`, `_find_syndrome`, `_apply_stabilizer`]
  - 結論:
    - 現在確認できる decoder 群はこの rotated CSS variant を前提にしている。

- XZZX:
  - 明示的記述:
    - `xzzx_code` という名前の参照はあるが、クラス本体・syndrome 規則・decoder entry point は確認できない。[根拠: `src/mcmc.py`, `Chain.update_chain_fast`]
  - 結論:
    - 「XXZZ と XZZX を共通 decoder で扱う」実装は、この repo 単体では確認できない。

### 3.6 高速化以外で重要な実装上の特徴

- EWD は X/Z を独立 matching しない。
  - 明示的記述:
    - `EWD_droplet_alpha` は chain 全体の `(n_x, n_y, n_z)` を数えて effective length を計算する。[根拠: `decoders.py`, `EWD_droplet_alpha`; `src/rotated_surface_model.py`, `RotSurCode.chain_lengths`]
  - 推論:
    - MWPM のような独立 X/Z matching graph ではなく、1 本の candidate Pauli chain を等価類単位で評価している。

- RNN 系も logical observable を直接 4-class softmax 予測していない。
  - 明示的記述:
    - ネットワーク出力は各 site の 2 値分布であり、最後に sample 群から `df_eq_distr` を後処理で作る。[根拠: `2drnn.py`, `MDRNNWavefunction.sample`, module `__main__`; `dilatedrnn.py`, `DilatedRNNWavefunction.sample`, module `__main__`]
  - 結論:
    - 直接 logical class classifier ではなく、candidate error strings の生成器として働いている。

## 4. 入出力インターフェースとデコードの運用形態

### 4.1 EWD パイプライン

- 入力データ:
  - 明示的記述:
    - decoder API の直接入力は `RotSurCode` オブジェクトであり、その内部に `qubit_matrix: np.uint8[size, size]` と `plaquette_defects: float[size+1, size+1]` を持つ。[根拠: `src/rotated_surface_model.py`, `RotSurCode.__init__`; `decoders.py`, `EWD_alpha`]
    - ベンチマーク用保存形式では、元の `qubit_matrix.astype(np.uint8)` を `df_qubit` として保存する。[根拠: `generate_data.py`, `generate`]

- syndrome の定義:
  - 明示的記述:
    - `plaquette_defects` の各要素は `_find_syndrome()` の defect bit であり、stabilizer 値そのものではなく「反可換な error が奇数個あるか」の 0/1 を表す。[根拠: `src/rotated_surface_model.py`, `_find_syndrome`]
    - syndrome は現在の data-qubit error 配置から直接計算され、round-to-round 差分は使わない。[根拠: `src/rotated_surface_model.py`, `RotSurCode.syndrome`]
  - 結論:
    - detection event ではなく、single-shot stabilizer defect。
    - 最終ラウンドを data-qubit readout から再構成する処理は無い。

- 出力データ:
  - 明示的記述:
    - `EWD_alpha()` は 4 等価類の百分率ベクトルを返す。[根拠: `decoders.py`, `EWD_alpha`]
    - `generate()` はそれを `df_eq_distr` として保存する。[根拠: `generate_data.py`, `generate`]
  - 結論:
    - 出力は「論理オブザーバブルの直接予測」に近い 4-class distribution であり、MWPM 用 edge weight でも exact correction string でもない。

- 運用形態:
  - 明示的記述:
    - `plot.py` は `argmax(eq_distr[:4])` を predicted logical class として評価する。[根拠: `plot.py`, module scope]
  - 結論:
    - logical readout-only decoding / logical memory 評価用途。
    - active correction として回路へフィードバックするインターフェースは未実装。

### 4.2 2D RNN VNA パイプライン

- 入力データ:
  - 明示的記述:
    - RNN の最適化対象 sample は `tf.int32` の `sampleplaceholder_forgrad` で shape `[numsamples, Nx, Ny]`。[根拠: `2drnn.py`, module `__main__`]
    - RNN cell 自体は `init_code.plaquette_defects` の近傍 4 値を条件付けに使う。[根拠: `2drnn.py`, `MDTensorizedRNNCell.call`]
  - 結論:
    - 入力は「2D binary tensor の候補 error string」と「固定された syndrome matrix」の組。

- syndrome の定義:
  - 明示的記述:
    - `failrate()` / `return_local_energies()` は、sample から作った `RotSurCode` の `plaquette_defects` を元 syndrome と比較する。[根拠: `2drnn.py`, `return_local_energies`, `failrate`]
  - 結論:
    - 各 bit は stabilizer defect bit。
    - detection event でも measurement-history でもない。

- 出力データ:
  - 明示的記述:
    - `MDRNNWavefunction.sample()` は `samples[num_samples, Nx, Ny]` と `log_probs` を返す。[根拠: `2drnn.py`, `MDRNNWavefunction.sample`]
    - 最終的には sample を `define_equivalence_class()` に通して `df_eq_distr` を集計する。[根拠: `2drnn.py`, module `__main__`]
  - 結論:
    - 内部出力は binary physical error strings。
    - 外部保存出力は 4 等価類の分布。

- 運用形態:
  - 明示的記述:
    - `failrate()` は syndrome 一致だけを判定し、最後の品質指標は等価類分布から logical class を読む。[根拠: `2drnn.py`, `failrate`, module `__main__`; `plot.py`, module scope]
  - 結論:
    - full correction string を外へ返す API は無い。
    - Pauli frame update 実装も無い。
    - logical readout-only decoding に分類するのが妥当。

### 4.3 1D dilated RNN VNA パイプライン

- 入力データ:
  - 明示的記述:
    - sample placeholder は `tf.int32` shape `[numsamples, N]`。[根拠: `dilatedrnn.py`, module `__main__`]
    - cell は `init_code.plaquette_defects` を参照しつつ flattened chain を逐次生成する。[根拠: `dilatedrnn.py`, `CustomRNNCell.call`, `DilatedRNNWavefunction.sample`]
  - 結論:
    - 入力表現は 1D binary chain。

- syndrome の定義:
  - 明示的記述:
    - `return_local_energies()` と `failrate()` は sample を `N_new x N_new` に reshape して syndrome を比較する。[根拠: `dilatedrnn.py`, `return_local_energies`, `failrate`]
  - 結論:
    - ここでも single-shot stabilizer defect を使う。

- 出力データ:
  - 明示的記述:
    - `DilatedRNNWavefunction.sample()` は flattened binary strings と log-probabilities を返す。[根拠: `dilatedrnn.py`, `DilatedRNNWavefunction.sample`]
    - 最終的には reshape 後に `define_equivalence_class()` で `df_eq_distr` を集計する。[根拠: `dilatedrnn.py`, module `__main__`]
  - 結論:
    - 内部出力は binary physical strings、外部保存出力は logical class distribution。

- 運用形態:
  - 結論:
    - 2D RNN と同様に logical readout-only decoding。
    - active correction / circuit feedback 可能な接口は未確認。

### 4.4 補足: 保存データ形式

- 明示的記述:
  - `generate_data.py`, `2drnn.py`, `dilatedrnn.py` はいずれも pandas DataFrame に `params`, `df_qubit`, `df_eq_distr` を順に保存する。[根拠: `generate_data.py`, `generate`; `2drnn.py`, module `__main__`; `dilatedrnn.py`, module `__main__`]
- 結論:
  - 既存 `.xz` artifacts は「入力 error matrix」と「decoder が出した 4-class distribution」の組を保持する評価用ファイルである。

## 5. Neural network 系アルゴリズムの対応

### 5.1 training と inference の対応状況

- 結論:
  - 2D RNN と 1D dilated RNN は、ともに「各 syndrome ごとにその場で最適化する」実装であり、通常の意味での offline training 済みモデル + inference-only パイプラインは確認できない。

- 明示的記述:
  - 両スクリプトとも optimizer, cost, `optstep` を定義し、各 datapoint ごとに `sess.run(initialize_parameters)` でパラメータを再初期化してから最適化ループに入る。[根拠: `2drnn.py`, module `__main__`; `dilatedrnn.py`, module `__main__`]
  - `saver = tf.compat.v1.train.Saver()` は作るが、保存済み checkpoint を読み込んで評価だけ行うコードは確認できない。[根拠: `2drnn.py`, module `__main__`; `dilatedrnn.py`, module `__main__`]
  - `training.py` にも同型の最適化ループがある。[根拠: `training.py`, module scope]

- 未確認:
  - `training.py` が現行パイプラインから再利用されている証拠は、今回確認した import 関係では見つからなかった。

- 推論:
  - この repo の neural 系は「decoder training infrastructure」よりも「per-instance variational optimization experiment」に近い。

### 5.2 合成訓練データの生成方法

- 結論:
  - supervised 用の labeled training dataset 生成パイプラインは確認できない。
  - 代わりに、各 datapoint の noisy code state をその場で生成し、その syndrome を使って RNN 分布を最適化している。

- 明示的記述:
  - `2drnn.py` は各 datapoint で `init_code = RotSurCode(N)` を作り、`generate_random_error(p_x, p_y, p_z)` で初期誤りを生成する。[根拠: `2drnn.py`, module `__main__`]
  - `dilatedrnn.py` も同様に `RotSurCode(N_new)` と `generate_random_error(...)` を使う。[根拠: `dilatedrnn.py`, module `__main__`]
  - 最適化の教師信号は label ではなく `return_local_energies()` が返す syndrome mismatch penalty と site-count penalty である。[根拠: `2drnn.py`, `return_local_energies`; `dilatedrnn.py`, `return_local_energies`]

- 推論:
  - `df_eq_distr` は訓練ラベルではなく、最適化後 sample 群から事後的に集計した推定分布である。

### 5.3 Neural 系の制約

- 明示的記述:
  - 出力 alphabet は 2 値 softmax で、4 値 Pauli を直接出力しない。[根拠: `2drnn.py`, `MDRNNWavefunction.__init__`; `dilatedrnn.py`, `DilatedRNNWavefunction.__init__`]
  - 実際の benchmark loop では `p_y = p_z = 0` で pure X noise を生成する。[根拠: `2drnn.py`, module `__main__`; `dilatedrnn.py`, module `__main__`]
- 結論:
  - neural 実装は本 repo で確認できる範囲では X-only の logical-class 推定器として使われている。

## 6. ベンチマークの評価内容

### 6.1 既存 plot / 評価スクリプトが測っているもの

- 結論:
  - `plot.py` が評価しているのは 1-shot readout ではなく、single-shot data-qubit error instance に対する logical class 推定失敗率である。

- 明示的記述:
  - `plot.py` は保存済み `qubit_matrix` から `true_eq = define_eq_rot(qubit_matrix)` を計算し、decoder 出力 `eq_distr[:4]` の `argmax` と比較して `failed/total` を `P_e` として描く。[根拠: `plot.py`, module scope]
  - 比較対象は `Rnn_test_entill2`, `Rnn_2dtest_d3`, `EWD_reference_size3` の 3 系統である。[根拠: `plot.py`, module scope]

- 推論:
  - 評価対象は logical memory / logical-class decoding であり、ancilla readout 単発分類ではない。

### 6.2 評価前提条件

- stabilizer rounds:
  - 明示的記述:
    - どの生成パイプラインも単一の `qubit_matrix` から syndrome を計算する。[根拠: `generate_data.py`, `generate`; `2drnn.py`, module `__main__`; `dilatedrnn.py`, module `__main__`; `src/rotated_surface_model.py`, `RotSurCode.syndrome`]
  - 結論:
    - repeated rounds なし。

- logical observable の偏り:
  - 明示的記述:
    - `_define_equivalence_class()` は 4 等価類を返し、`plot.py` も `eq_distr[:4]` 全体の `argmax` を比較する。[根拠: `src/rotated_surface_model.py`, `_define_equivalence_class`; `plot.py`, module scope]
  - 結論:
    - logical Z 限定ではなく、4 class 全体の識別評価。

- 誤りモデル:
  - EWD 系:
    - 明示的記述:
      - `run.py` は `noise='alpha'` を設定する。[根拠: `run.py`, module `__main__`]
      - ただし実際の注入は `generate_data.py` の `alpha` 分岐で `p_y = p_z = 0` に上書きされる。[根拠: `generate_data.py`, `get_individual_error_rates`]
    - 結論:
      - 実質 pure X code-capacity。
  - RNN 系:
    - 明示的記述:
      - `2drnn.py`, `dilatedrnn.py` とも `p_x = ps[p_idx]`, `p_y = p_z = 0` で生成する。[根拠: `2drnn.py`, module `__main__`; `dilatedrnn.py`, module `__main__`]
    - 結論:
      - 実質 pure X code-capacity。

### 6.3 結果ファイルの存在範囲

- 明示的記述:
  - `data/` には `EWD_reference_size3_*`, `EWD_reference_size5_*`, `Rnn_2dtest_d3_*`, `Rnn_2dtest_d5_*`, `Timecheck_2D_size5_0.xz`, `Timecheck_ewddecoder_size5_0.xz` などの既存 artifact がある。[根拠: `data/` 配下ファイル名, 関数/クラス名なし]
  - `plot.py` が直接使うのは d=3 系列のみ。[根拠: `plot.py`, module scope]
- 結論:
  - d=3 の logical error rate plot は repo 内スクリプトで直接再現対象と読める。
  - d=5 artifact は存在するが、同梱の plot script で直接可視化されているかは未確認。

### 6.4 しきい値図・circuit-level 評価

- しきい値図:
  - 未確認。
  - 理由:
    - 物理誤り率 sweep 用の結果ファイルはあるが、distance 複数本を系統的に重ねて threshold を算出するスクリプトは repo 内で確認できない。

- circuit-level / measurement-fault benchmark:
  - 結論: 非対応。
  - 根拠:
    - すべての評価スクリプトが single-shot data-qubit error 配列だけを生成・評価している。[根拠: `generate_data.py`, `generate`; `2drnn.py`, module `__main__`; `dilatedrnn.py`, module `__main__`]
