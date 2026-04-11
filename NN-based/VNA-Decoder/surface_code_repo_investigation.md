# Repo Investigation: Matching from Syndrome Graph in `VNA-Decoder`

## 0. Executive Summary

- 結論: この checkout にある実装は、シンドロームグラフ上のマッチング処理を実装していない。
  - 明示的記述:
    - `README.md` は「variational neural annealing with RNNs」と「EWD decoder as a reference decoder」だけを述べており、MWPM / Blossom / PyMatching の記述がない。[根拠: `README.md`, module scope]
    - `generate_data.py` は decoder entry point として `EWD_alpha` を import し、`params['method'] == "EWD"` ではそれを呼ぶ。[根拠: `generate_data.py`, module scope; `generate_data.py`, `generate`]
    - `decoders.py` の実装は `Chain_alpha` を使ったサンプリングと等価類分布 `eqdistr` の計算であり、グラフ構築・最短路・完全マッチングの API を持たない。[根拠: `decoders.py`, `EWD_alpha`, `EWD_droplet_alpha`]
    - `requirements.txt` には `numba`, `pandas`, `scipy`, `matplotlib` しかなく、グラフ matching 系依存が確認できない。[根拠: `requirements.txt`, module scope]
  - 推論:
    - 実装の中心は「qubit 配列を直接更新する MCMC/EWD」と「syndrome 条件付きで qubit 配列をサンプルする VNA-RNN」であり、ユーザ要求の「シンドロームグラフからマッチングを出力する過程」は repo の主経路としては存在しない。

- 例外的な未完了フック:
  - 明示的記述:
    - `generate_data.py` に `params['mwpm_init']` と `class_sorted_mwpm(init_code)` の呼び出しがある。[根拠: `generate_data.py`, `generate`]
    - `decoders.py` に `class_sorted_mwpm(init_code)` / `regular_mwpm(init_code)` のコメントアウトが残っている。[根拠: `decoders.py`, module `__main__`]
    - しかし、これらの関数定義は repo 内で確認できない。[根拠: repo 全体検索 `rg "class_sorted_mwpm|regular_mwpm"`]
  - 推論:
    - MWPM を導入しようとした痕跡はあるが、この checkout 単体では未実装または欠落状態であり、「ラップしている」とは言えない。

## 1. 対象となるグラフ構造と実装範囲

### 1.1 シンドロームグラフ／matching 実装の有無

- 結論: 実装済みの matching graph solver は未確認であり、実動コードとしては「未対応」。
  - 明示的記述:
    - `RotSurCode` は `qubit_matrix` と `plaquette_defects` を保持するが、隣接リスト・辺集合・ノード ID 配列のような graph object を持たない。[根拠: `src/rotated_surface_model.py`, `RotSurCode.__init__`]
    - `RotSurCode.syndrome()` は `plaquette_defects` を直接埋めるだけで、graph 構築処理を呼ばない。[根拠: `src/rotated_surface_model.py`, `RotSurCode.syndrome`]
    - `EWD_alpha` は `Chain_alpha` を各等価類に対して回し、`eqdistr` を返す。[根拠: `decoders.py`, `EWD_alpha`]
  - 推論:
    - この repo の内部表現は「検出イベントのグラフ」ではなく「データ qubit の Pauli 配列」と「そこから再計算される 2D syndrome 配列」である。

### 1.2 実際に確認できる surface-code 幾何

- 結論: 実装本体として確認できるのは、open-boundary の 2D rotated planar CSS patch だけである。
  - 明示的記述:
    - `RotSurCode.__init__` は `qubit_matrix.shape == (size, size)` と `plaquette_defects.shape == (size + 1, size + 1)` を用いる。[根拠: `src/rotated_surface_model.py`, `RotSurCode.__init__`]
    - `RotSurCode.syndrome()` は interior plaquette と 4 辺の half plaquette を計算する。[根拠: `src/rotated_surface_model.py`, `RotSurCode.syndrome`]
    - `_find_syndrome(..., operator=1)` は parity に応じて `op = 1` / `op = 3` を交互配置し、`operator=3` では 4 辺の half stabilizer を使う。[根拠: `src/rotated_surface_model.py`, `_find_syndrome`]
  - 推論:
    - toric code 由来の周期境界グラフ、3D time-like edge を持つ detection graph、XZZX 専用の別 graph 実装は、この checkout では確認できない。

### 1.3 3D 対応・境界ノード・制約

- 2D / 3D:
  - 明示的記述:
    - syndrome は 2 次元配列 `plaquette_defects` 1 枚だけで、時間方向 index がない。[根拠: `src/rotated_surface_model.py`, `RotSurCode.__init__`, `RotSurCode.syndrome`]
  - 結論:
    - 2D code-capacity 相当のみ。3D 時空間グラフは未対応。

- 境界ノード:
  - 明示的記述:
    - 境界は half plaquette として `plaquette_defects` の外周セルに埋め込まれる。[根拠: `src/rotated_surface_model.py`, `RotSurCode.syndrome`; `src/rotated_surface_model.py`, `_find_syndrome`]
  - 推論:
    - open boundary 自体は syndrome 配列に表現されるが、matching solver が使う「boundary node」という離散 graph node は定義されていない。

### 1.4 Capability Matrix

| 対象 | Graph dimension (2D/3D) | Boundary node support | Weighted edges support | Hyperedge support (for correlated/Y errors) | Dynamic graph generation | Parallel matching support |
| --- | --- | --- | --- | --- | --- | --- |
| 実装済み EWD/VNA 経路 | 2D の syndrome 配列のみ | open boundary は half plaquette として表現、明示的 boundary node はなし | なし | なし | 明示的 graph 生成なし。`plaquette_defects` を都度再計算 | なし |
| `mwpm_init` フック | 未確認 | 未確認 | 未確認 | 未確認 | 未確認 | 未確認 |

- `mwpm_init` 行の判定理由:
  - 明示的記述:
    - `generate_data.py` の `mwpm_init` 分岐は `class_sorted_mwpm(init_code)` を呼ぶが、定義元が repo 内で見つからない。[根拠: `generate_data.py`, `generate`; repo 全体検索 `rg "class_sorted_mwpm"`]
  - 結論:
    - 実装欠落のため capability 判定は「未確認」。

## 2. グラフ構築とエッジ重みの計算

### 2.1 シンドロームグラフ構築の有無

- 結論: syndrome graph の構築処理は未確認。
  - 明示的記述:
    - `RotSurCode.syndrome()` は各 plaquette defect bit を `self.plaquette_defects[...]` に代入するのみで、graph 構造を生成しない。[根拠: `src/rotated_surface_model.py`, `RotSurCode.syndrome`]
    - `_find_syndrome()` は qubit 4 点または 2 点の parity から 0/1 defect を返すだけで、edge を返さない。[根拠: `src/rotated_surface_model.py`, `_find_syndrome`]

### 2.2 Code-capacity と phenomenological / circuit-level の差

- 結論: 実装されているのは code-capacity 由来の 2D 単発 syndrome のみ。
  - 明示的記述:
    - `generate_random_error(p_x, p_y, p_z)` は data-qubit 配列 `qubit_matrix` に Pauli 値を入れてから `self.syndrome()` を 1 回呼ぶ。[根拠: `src/rotated_surface_model.py`, `RotSurCode.generate_random_error`]
    - `generate_data.py` は `RotSurCode(params['size'])` を作り、1 回の `generate_random_error(...)` を用いてデータを作る。[根拠: `generate_data.py`, `generate`]
  - 推論:
    - 測定誤り・繰り返し測定・time-like edge を持つ graph は設計対象外であり、「未実装」よりも「現在の表現が 2D 単発 syndrome に閉じている」と記述する方が正確。

### 2.3 エッジ重みの代わりに実際に使われる量

- 結論: edge weight は保持されず、EWD では chain 全体の effective length が重みの代用品として使われる。
  - 明示的記述:
    - `chain.code.chain_lengths()` は `(nx, ny, nz)` を返す。[根拠: `src/rotated_surface_model.py`, `RotSurCode.chain_lengths`]
    - `EWD_droplet_alpha()` は `eff_len = lengths[2] + alpha * sum(lengths[0:2])` を計算する。[根拠: `decoders.py`, `EWD_droplet_alpha`]
    - `EWD_alpha()` は `beta = -np.log(pz_tilde)` として `exp(-beta * eff_len)` を等価類ごとに足し上げる。[根拠: `decoders.py`, `EWD_alpha`]
    - `Chain_alpha.update_chain_fast()` は `_apply_random_stabilizer(...)` で新しい qubit 配列を提案し、`pz_tilde**(dz + alpha*(dx + dy))` に基づいて受理する。[根拠: `src/mcmc_alpha.py`, `Chain_alpha.update_chain_fast`, `_update_chain_fast_rotated`]
  - 推論:
    - これは graph edge 単位の cost ではなく、candidate error chain 全体に対する Boltzmann-like weight である。

### 2.4 相関誤りと Y 誤り

- 基本表現:
  - 明示的記述:
    - `generate_random_error()` 自体は `p_x`, `p_y`, `p_z` を受け取り、`qubit_matrix` に 0/1/2/3 を入れる。[根拠: `src/rotated_surface_model.py`, `RotSurCode.generate_random_error`]
  - 推論:
    - 基底となる qubit 配列表現は X/Y/Z を区別できる。

- 実際の benchmark 経路:
  - 明示的記述:
    - `get_individual_error_rates()` の `alpha` 分岐は一度 `p_x, p_y, p_z` を計算した後、`p_y = p_z = 0` に上書きする。[根拠: `generate_data.py`, `get_individual_error_rates`]
    - `generate_data.py` の `EWD` 分岐は `noise == 'alpha'` 以外を拒否する。[根拠: `generate_data.py`, `generate`]
  - 結論:
    - 現行の EWD ベンチマークは pure-X noise であり、Y 相関に対応する hyperedge matching は使われない。

- X/Z 分離 graph の有無:
  - 明示的記述:
    - `_find_syndrome()` は X/Z plaquette を 1 枚の `plaquette_defects` 配列に交互配置する。[根拠: `src/rotated_surface_model.py`, `_find_syndrome`]
  - 推論:
    - MWPM で典型的な「X graph と Z graph を分離して解く」経路は、この repo では確認できない。

## 3. マッチングアルゴリズムの概要

### 3.1 実装済み core algorithm

- 結論: graph matching solver ではなく、以下の 2 系統がある。

- EWD / MCMC 系:
  - 明示的記述:
    - `EWD_alpha()` は 4 つの logical equivalence class ごとに `Chain_alpha(copy.deepcopy(init_code), ...)` を作る。[根拠: `decoders.py`, `EWD_alpha`]
    - 各 chain は `to_class(eq)` で同 syndrome の別等価類に移り、`update_chain_fast()` で stabilizer move を繰り返す。[根拠: `decoders.py`, `EWD_alpha`; `src/rotated_surface_model.py`, `RotSurCode.to_class`; `src/mcmc_alpha.py`, `Chain_alpha.update_chain_fast`]
  - 推論:
    - これは MWPM / Union-Find / Blossom ではなく、stabilizer 更新に基づくサンプリング decoder である。

- VNA-RNN 系:
  - 明示的記述:
    - `2drnn.py` の `MDRNNWavefunction.sample()` は `samples` を生成する。[根拠: `2drnn.py`, `MDRNNWavefunction.sample`]
    - `2drnn.py` の `return_local_energies()` は sample を `RotSurCode.update_matrix()` に渡し、target syndrome との差と `np.sum(samples)` を energy にする。[根拠: `2drnn.py`, `return_local_energies`]
    - `dilatedrnn.py` でも同様に flattened sample を `N_new x N_new` に reshape して同じ種の energy を作る。[根拠: `dilatedrnn.py`, `return_local_energies`]
  - 推論:
    - これらは matching edge を出力する NN ではなく、syndrome 条件付きの qubit-pattern sampler / optimizer である。

### 3.2 外部ライブラリ依存

- 結論: matching solver への依存は未確認。
  - 明示的記述:
    - repo 検索では `PyMatching`, `Blossom`, `networkx`, `Matching` の import / 実装が見つからない。[根拠: repo 全体検索 `rg "PyMatching|Blossom|networkx|Matching"`]
    - `README.md` は EWD と VNA-RNN だけを説明する。[根拠: `README.md`, module scope]

### 3.3 `src/mcmc.py` に残る未解決参照

- 明示的記述:
  - `Chain.update_chain_fast()` は `xzzx_code`, `Planar_code`, `Toric_code` とそれぞれの高速更新関数を参照する。[根拠: `src/mcmc.py`, `Chain.update_chain_fast`]
- 未確認:
  - これらのクラスや更新関数の定義は、この repo では確認できない。[根拠: repo 全体検索 `rg "class xzzx_code|class Planar_code|class Toric_code|_update_chain_fast_xzzx|_update_chain_fast_planar|_update_chain_fast_toric"`]
- 推論:
  - 旧 repo 由来コードの残骸、または欠落ファイルを前提にした未完成状態の可能性がある。少なくとも現 checkout で matching support を主張する根拠にはならない。

## 4. 入出力インターフェースとデータ構造

### 4.1 入力データ（graph）

- 結論: solver に渡す explicit graph は存在しない。
  - 明示的記述:
    - `RotSurCode` の主要状態は `qubit_matrix` と `plaquette_defects` の 2 配列だけである。[根拠: `src/rotated_surface_model.py`, `RotSurCode.__init__`]
    - `generate_data.py` は `init_code: RotSurCode` をそのまま `EWD_alpha()` に渡す。[根拠: `generate_data.py`, `generate`; `decoders.py`, `EWD_alpha`]

### 4.2 入力データ（syndrome / defects）

- EWD:
  - 明示的記述:
    - `EWD_alpha()` の引数は `init_code` であり、defect node の list を別引数では受け取らない。[根拠: `decoders.py`, `EWD_alpha`]
  - 推論:
    - syndrome は `init_code.plaquette_defects` として object 内に暗黙保持される。

- VNA 2D:
  - 明示的記述:
    - `MDTensorizedRNNCell.call()` は `init_code.plaquette_defects` の近傍 4 値を constant tensor として読み込む。[根拠: `2drnn.py`, `MDTensorizedRNNCell.call`]
    - `return_local_energies(samples, init_code)` は `samples[x, :, :]` を `RotSurCode.update_matrix()` に渡して syndrome を比較する。[根拠: `2drnn.py`, `return_local_energies`]

- VNA 1D dilated:
  - 明示的記述:
    - `CustomRNNCell.call()` は `init_code.plaquette_defects` の近傍 4 値を用いる。[根拠: `dilatedrnn.py`, `CustomRNNCell.call`]
    - `return_local_energies(samples, init_code)` は flattened sample を `reshape(N_new, N_new)` 後に `update_matrix()` へ渡す。[根拠: `dilatedrnn.py`, `return_local_energies`]

### 4.3 出力データ

- 結論: matching pair list や selected edge list は返さない。返すのは logical equivalence class 分布である。
  - 明示的記述:
    - `EWD_alpha()` は `np.divide(eqdistr, sum(eqdistr)) * 100` を返す。[根拠: `decoders.py`, `EWD_alpha`]
    - `2drnn.py` は sampled matrix ごとに `define_equivalence_class()` を数えて `df_eq_distr` を作る。[根拠: `2drnn.py`, module `__main__`]
    - `dilatedrnn.py` も同様に `df_eq_distr` を作る。[根拠: `dilatedrnn.py`, module `__main__`]

### 4.4 コア関数・クラスのシグネチャ

- `RotSurCode(size)`:
  - 役割: data-qubit 配列と syndrome 配列の保持・更新。
  - 主要メソッド: `generate_random_error(p_x, p_y, p_z)`, `syndrome()`, `update_matrix(newmatrix)`, `define_equivalence_class()`, `to_class(eq)`。[根拠: `src/rotated_surface_model.py`, `RotSurCode`]

- `EWD_alpha(init_code, pz_tilde, alpha, steps, pz_tilde_sampling=None, onlyshortest=True)`:
  - 入力: `RotSurCode` と scalar parameter 群。
  - 出力: 4 要素の等価類分布 `np.ndarray` 相当。[根拠: `decoders.py`, `EWD_alpha`]

- `return_local_energies(samples, init_code)`:
  - 2D RNN 版入力: `samples.shape == (numsamples, Nx, Ny)`。[根拠: `2drnn.py`, `return_local_energies`]
  - 1D dilated 版入力: `samples.shape == (numsamples, N_new * N_new)` を reshape して使う。[根拠: `dilatedrnn.py`, `return_local_energies`]
  - 出力: sample ごとの local energy の 1D array。[根拠: `2drnn.py`, `return_local_energies`; `dilatedrnn.py`, `return_local_energies`]

## 5. Neural network 系アルゴリズムの適用

- 結論: NN は edge weight 推論にも matching edge 直接予測にも使われていない。
  - 明示的記述:
    - `2drnn.py` / `dilatedrnn.py` は `plaquette_defects` を局所条件として用いながら binary sample を生成する。[根拠: `2drnn.py`, `MDTensorizedRNNCell.call`, `MDRNNWavefunction.sample`; `dilatedrnn.py`, `CustomRNNCell.call`, `DilatedRNNWavefunction.sample`]
    - 学習・最適化の target は `return_local_energies()` が返す syndrome mismatch penalty と `np.sum(samples)` である。[根拠: `2drnn.py`, `return_local_energies`; `dilatedrnn.py`, `return_local_energies`]
  - 推論:
    - NN の出力は「候補 correction chain の qubit occupancy」であり、「どの defect とどの defect を結ぶか」という matching edge prediction ではない。

- テンソル化の形:
  - 2D RNN:
    - `samples` は `(numsamples, Nx, Ny)` binary tensor。[根拠: `2drnn.py`, `return_local_energies`, `MDRNNWavefunction.sample`]
  - 1D dilated RNN:
    - `samples` は flatten された binary vector で、energy 計算時に `reshape(N_new, N_new)` される。[根拠: `dilatedrnn.py`, `return_local_energies`]

## 6. パフォーマンス・ベンチマーク

- repo が実際に評価している主指標:
  - 明示的記述:
    - `plot.py` は pickle から `qubit_matrix` と `eq_distr` を読み、`true_eq` と `argmax(eq_distr[:4])` を比較して `P_e` を計算する。[根拠: `plot.py`, module scope]
  - 結論:
    - 主評価指標は logical error rate (`P_e` / `P_f`) であり、matching solver としての性能ではない。

- VNA 内部の補助指標:
  - 明示的記述:
    - `2drnn.py` / `dilatedrnn.py` の `failrate()` は、sampled qubit matrix が target syndrome と一致しない割合を返す。[根拠: `2drnn.py`, `failrate`; `dilatedrnn.py`, `failrate`]
  - 推論:
    - これは最適化ループの収束監視であり、matching runtime・memory・cost gap の benchmark ではない。

- 未確認事項:
  - matching 実行時間
  - graph size / syndrome density に対する scaling
  - memory usage
  - 厳密解との matching cost 差
  - 理由:
    - `run.py` は `generate(...)` を回すだけで、保存するのは等価類分布データである。[根拠: `run.py`, module `__main__`; `generate_data.py`, `generate`]
    - `2drnn.py` / `dilatedrnn.py` には `start = time.time()` があるが、今回確認した範囲ではその値を集計・保存・可視化するコードは見つからない。[根拠: `2drnn.py`, module `__main__`; `dilatedrnn.py`, module `__main__`]

## 7. Final Assessment

- この repo を「シンドロームグラフからマッチング結果を返す decoder 実装」として読むのは不正確である。
- 正確には、現在の checkout は以下の 2 系統を持つ:
  - `RotSurCode` 上の qubit 配列を stabilizer move で探索する EWD/MCMC 系。
  - `plaquette_defects` を条件に qubit 配列を直接サンプルする VNA-RNN 系。
- したがって、本調査項目のうち graph matching に固有な項目は多くが「未対応」または「未確認」であり、これは単に記述漏れではなく、現在の主実装が別種の decoder を採っていることに由来する。
