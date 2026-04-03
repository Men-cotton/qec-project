# Astra repo 調査メモ: surface code / noise model / decoder / benchmark

この文書は、repo 調査結果を段階ごとに追記する作業ログである。各節では、コードや README にある「明示的記述」と、そこから導ける「推論」を分けて書く。証拠が見つからない項目は「未確認」とする。

## 1. Surface code の対応状況と実装範囲

### 1.1 結論

- この repo は surface code の QEC に対応している。
  - 主対象ファイル: `README.md`, `gnn_train.py`, `gnn_test.py`, `gnn_osd.py`, `panq_functions.py`
  - 補助的な surface-code 生成関数: `codes_q.py`

### 1.2 対応の根拠

#### 明示的記述

- `README.md` は「surface codes up to distance 9 affected by code capacity noise」と明記している。
  - 根拠: `README.md` / 本文
- 学習・評価スクリプトはいずれも `surface_2d.RotatedPlanar2DCode(dist)` を生成している。
  - 根拠: `gnn_train.py` / top-level script
  - 根拠: `gnn_test.py` / top-level script
  - 根拠: `gnn_osd.py` / top-level script
- `codes_q.py` には surface code 系の生成関数 `create_surface_codes`, `create_rotated_surface_codes`, `create_checkerboard_toric_codes` が実装されている。
  - 根拠: `codes_q.py` / `create_surface_codes`
  - 根拠: `codes_q.py` / `create_rotated_surface_codes`
  - 根拠: `codes_q.py` / `create_checkerboard_toric_codes`

#### 推論

- 実運用上の surface-code デコード対象は、repo 内の主要パイプラインでは `panqec.codes.surface_2d.RotatedPlanar2DCode` に限定される。
  - 理由: `gnn_train.py`, `gnn_test.py`, `gnn_osd.py` のいずれもこのクラスを直接生成しており、`codes_q.py` 側の surface-code 生成関数はコメントアウト以外で参照されていない。
  - 根拠: `gnn_train.py` / top-level script
  - 根拠: `gnn_test.py` / top-level script
  - 根拠: `gnn_osd.py` / top-level script
  - 根拠: `codes_q.py` / `create_surface_codes`, `create_rotated_surface_codes`, `create_checkerboard_toric_codes`
- main pipeline が扱うのは X/Z を分離した CSS 型 rotated planar surface code であり、XZZX の実装証拠は見つからない。
  - 理由: repo 内の surface-code 関連ロジックは `code.Hx`, `code.Hz`, `code.logicals_x`, `code.logicals_z`, `code.measure_syndrome`, `code.logical_errors` を使って X/Z を分けて扱っている。`XZZX` という識別子や混成 stabilizer 定義は repo 内に存在しない。
  - 根拠: `panq_functions.py` / `surface_code_edges`, `logical_error_rate`, `osd`, `ler_loss`
  - 根拠: `gnn_train.py` / top-level script
  - 根拠: `gnn_test.py` / top-level script
  - 根拠: `gnn_osd.py` / `logical_error_rate_osd`, `osd`

### 1.3 実装されているコード種別

#### 1.3.1 main pipeline で実際に使われるコード

- `RotatedPlanar2DCode`
  - 明示的記述:
    - 学習・評価・GNN+2段目デコーダの各スクリプトが `surface_2d.RotatedPlanar2DCode(dist)` を使う。
    - 根拠: `gnn_train.py` / top-level script
    - 根拠: `gnn_test.py` / top-level script
    - 根拠: `gnn_osd.py` / top-level script
  - 推論:
    - 実装スタイルは X と Z を分離した rotated planar CSS surface code であり、少なくとも repo 内には XZZX 専用処理はない。
    - 根拠: `panq_functions.py` / `surface_code_edges`, `generate_syndrome_error_volume`, `logical_error_rate`, `osd`

#### 1.3.2 repo 内に存在するが main pipeline に未統合の生成関数

- `create_rotated_surface_codes(n)`
  - 明示的記述:
    - odd `n` を強制し、`hx` と `hz` を別々に構成する rotated surface code 生成関数。
    - 根拠: `codes_q.py` / `create_rotated_surface_codes`
  - 推論:
    - これも X/Z 分離の CSS rotated code であり、XZZX ではない。
    - 根拠: `codes_q.py` / `create_rotated_surface_codes`
- `create_surface_codes(n)`
  - 明示的記述:
    - repetition code の hypergraph product で `[n^2 + (n-1)^2, 1, n] surface code` を生成する。
    - 根拠: `codes_q.py` / `create_surface_codes`
  - 未確認:
    - この関数を用いたベンチマークやデコーダ接続は repo 内で確認できない。
- `create_checkerboard_toric_codes(n)`
  - 明示的記述:
    - even `n` を強制する checkerboard toric code 生成関数。
    - 根拠: `codes_q.py` / `create_checkerboard_toric_codes`
  - 注意:
    - toric code は surface-code family の一種として生成関数はあるが、main pipeline の対象ではない。

### 1.4 実装上の制約

#### 明示的記述

- `create_rotated_surface_codes(n)` は odd `n` のみ許可する。
  - 根拠: `codes_q.py` / `create_rotated_surface_codes`
- main pipeline は `dist` という単一スカラーを入力に取り、サイズ計算を `d^2`, `2*d^2`, `2*d^2-1` に固定している。
  - 根拠: `gnn_train.py` / top-level script
  - 根拠: `gnn_test.py` / top-level script
  - 根拠: `panq_functions.py` / `surface_code_edges`, `generate_syndrome_error_volume`, `adapt_trainset`, `logical_error_rate`, `osd`, `ler_loss`
- `create_rotated_surface_codes(n)` は境界の weight-2 check を明示的に追加している。
  - 根拠: `codes_q.py` / `create_rotated_surface_codes`

#### 推論

- main pipeline の surface-code 実装は square patch 前提であり、rectangular patch を直接受けるインターフェースはない。
  - 理由: サイズ計算が全面的に `d^2` ベースで固定されている。
  - 根拠: `gnn_train.py` / top-level script
  - 根拠: `gnn_test.py` / top-level script
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`, `adapt_trainset`, `logical_error_rate`
- main pipeline の対象は single logical qubit の rotated planar code である可能性が高い。
  - 理由: `RotatedPlanar2DCode` というクラス名、および `d^2` 固定のパラメタ化から複数 logical patch を並べる設計には見えない。
  - 根拠: `gnn_train.py` / top-level script
  - 根拠: `gnn_test.py` / top-level script
  - 判定: 推論
- main pipeline は open boundary の planar patch を想定している可能性が高い。
  - 理由: 使用クラス名が `RotatedPlanar2DCode` であり、repo 内独自 rotated code 生成でも境界 check を明示追加している。
  - 根拠: `gnn_train.py` / top-level script
  - 根拠: `codes_q.py` / `create_rotated_surface_codes`
  - 判定: 推論
- repo 内に XZZX, rectangular rotated patch, lattice surgery 用 patch composition の明示実装は見つからない。
  - 根拠: repo 全文検索結果
  - 判定: 未実装か対象外かは未確定。少なくとも repo 内に明示証拠はない。

### 1.5 Capability Matrix

| code family | patch shape | single/multi logical qubit | boundaries | odd distance restriction | repeated syndrome rounds | measurement error support | active correction support | lattice surgery | benchmark scripts | neural decoder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rotated planar 2D (main pipeline; PanQEC `RotatedPlanar2DCode`) | square のみと推定 | single logical と推定 | open planar と推定 | 未確認 | なし | なし | なし | なし | あり (`gnn_test.py`, `gnn_osd.py`) | あり (`GNNDecoder`) |
| Rotated surface constructor (`create_rotated_surface_codes`) | square | 未確認 | 境界 check あり | あり (odd `n` 限定) | なし | なし | なし | なし | なし | なし |
| Hypergraph-product surface constructor (`create_surface_codes`) | square | コメント上は `k=1` | 未確認 | 未確認 | なし | なし | なし | なし | なし | なし |
| Checkerboard toric constructor (`create_checkerboard_toric_codes`) | square periodic | toric なので multi logical の可能性が高いが repo 内では未確認 | periodic と推定 | even `n` 限定 | なし | なし | なし | なし | なし | なし |

### 1.6 未対応機能の整理

- XZZX surface code
  - 状態: 未確認
  - 理由の切り分け:
    - 「単に実装が見当たらない」の証拠はある。repo 内検索で `XZZX` 識別子や混合 stabilizer 実装が見つからない。
    - ただし「設計上対象外」と明記した文書は見つかっていない。
- rectangular patch
  - 状態: main pipeline では未対応と推定
  - 理由の切り分け:
    - `d^2` 固定のデータ表現から、少なくとも現状実装は square-only 前提に見える。
    - 「対象外」と明記した文書は未確認。
- lattice surgery
  - 状態: 未対応
  - 理由の切り分け:
    - 実装・スクリプト・README のいずれにも対応箇所が見つからない。
    - patch merge/split, multi-patch scheduling, joint stabilizer measurement などの痕跡も repo 内で未確認。

## 2. 対象ノイズモデル

### 2.1 結論

- surface-code main pipeline が前提にしているのは、基本的に **code capacity / single-shot syndrome** である。
- measurement error を含む phenomenological / circuit-level ノイズの実装証拠は repo 内に見つからない。
- 物理 Pauli 誤りとしては X/Z に加えて Y も表現される。
- ただし MWPM/OSD との接続では、Y は独立な X 成分と Z 成分に分解して扱われており、相関誤りを 1 つの相関付き matching graph として扱う実装は確認できない。

### 2.2 デコーダ実装が仮定するノイズモデル

#### 明示的記述

- `generate_syndrome_error_volume` は 1 回の `error_model.generate(code, p)` からエラーを生成し、`code.measure_syndrome(error).T` で単発 syndrome を得ている。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`
- 入力データ構造には時間方向のラウンド軸がなく、サイズは `2*d^2 - 1` に固定される。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`, `adapt_trainset`, `logical_error_rate`, `osd`
- 物理誤りラベルは `error[:,:d**2] + 2*error[:,d**2:]` で 0/1/2/3 にエンコードされる。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`
- GNN 出力後の 2段目 matching では、`fllrx` と `fllrz` を別々に作り、`MatchingDecoder(..., weights=(fllrx[i], fllrz[i]))` に渡している。
  - 根拠: `panq_functions.py` / `osd`
- GNN+OSD 系でも `BeliefPropagationOSDDecoder` の `x_decoder` と `z_decoder` に別々の log-prob を設定している。
  - 根拠: `gnn_osd.py` / `osd`, `init_log_probs_of_decoder`

#### 推論

- GNN 単体パイプラインが仮定しているのは **perfect measurement を持つ code-capacity ノイズ** である可能性が高い。
  - 理由: 単発 syndrome しか入力しておらず、measurement error 用の時間軸や検出イベント差分が存在しない。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`, `adapt_trainset`
- MWPM/OSD 接続部は、**独立した X/Z 成分への分解** を前提にしている。
  - 理由: syndrome も物理誤りも X 成分と Z 成分に分割して処理し、matching/OSD の重みも `fllrx`, `fllrz` に分離される。
  - 根拠: `panq_functions.py` / `logical_error_rate`, `osd`
  - 根拠: `gnn_osd.py` / `logical_error_rate_osd`, `osd`
- Y 誤りは物理ラベルとしては表現されるが、matching/OSD 段では X/Z の同時発生へ分解される。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`, `logical_error_rate`, `osd`

### 2.3 ベンチマーク / シミュレーションスクリプトが実際に使うノイズモデル

#### 明示的記述

- README は評価対象を「surface codes ... affected by code capacity noise」と記述している。
  - 根拠: `README.md` / 本文
- `gnn_train.py`, `gnn_test.py`, `gnn_osd.py`, `panq_nvidia.py` はいずれも `error_model_name` として `X`, `Z`, `XZ`, `DP` を持ち、既定値は `DP` である。
  - 根拠: `gnn_train.py` / top-level script
  - 根拠: `gnn_test.py` / top-level script
  - 根拠: `gnn_osd.py` / top-level script
  - 根拠: `panq_nvidia.py` / top-level script
- `DP` は `PauliErrorModel(0.34, 0.32, 0.34)` に設定されている。
  - 根拠: `gnn_train.py` / top-level script
  - 根拠: `gnn_test.py` / top-level script
  - 根拠: `gnn_osd.py` / top-level script
  - 根拠: `panq_nvidia.py` / top-level script
- テスト/評価データは `generate_syndrome_error_volume(..., for_training=False)` で固定 `p` のサンプルを生成する。
  - 根拠: `gnn_test.py` / top-level script
  - 根拠: `gnn_osd.py` / top-level script
  - 根拠: `panq_nvidia.py` / top-level script
- 学習データは `for_training=True` で生成され、各サンプルの実効誤り率は `pr = p * rng.random()` で一様にばらつかせている。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`

#### 推論

- 実際の benchmark / simulation は **measurement error なしの code-capacity Pauli noise** を使っている。
  - 理由: 生成されるのは単発の data-qubit Pauli error と単発 syndrome のみで、測定誤りを加える処理がない。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`
  - 根拠: `README.md` / 本文
- `X`, `Z`, `XZ`, `DP` の切替により、偏った X-only, Z-only, X/Z 対称, depolarizing-like ノイズは扱える。
  - 根拠: `gnn_train.py` / top-level script
  - 根拠: `gnn_test.py` / top-level script
  - 根拠: `gnn_osd.py` / top-level script
- 一般の measurement-biased phenomenological noise や circuit-level hook error は未対応である可能性が高い。
  - 理由: measurement channel, gate schedule, repeated rounds, detector graph 構築のいずれも repo 内に見当たらない。
  - 判定: 推論

### 2.4 要素別整理

| 観点 | デコーダ実装の前提 | benchmark / simulation で実使用 | 根拠 |
| --- | --- | --- | --- |
| 前提モデル分類 | code capacity と推定 | code capacity | `README.md` 本文; `panq_functions.py` / `generate_syndrome_error_volume` |
| measurement error | なし | なし | `panq_functions.py` / `generate_syndrome_error_volume`, `adapt_trainset` |
| data error only | はい | はい | `panq_functions.py` / `generate_syndrome_error_volume` |
| Y error | あり | あり (`DP`) | `panq_functions.py` / `generate_syndrome_error_volume`; `gnn_train.py` などの `PauliErrorModel(0.34, 0.32, 0.34)` |
| 相関誤り | GNN 出力には Y として現れるが、matching/OSD 段では X/Z 分解 | 同左 | `panq_functions.py` / `osd`; `gnn_osd.py` / `osd` |
| biased noise | `X`, `Z`, `XZ` などで bias を変えられる | 既定値は `DP`; 他設定もコード上は可能 | `gnn_train.py`, `gnn_test.py`, `gnn_osd.py`, `panq_nvidia.py` / top-level script |
| phenomenological | 未確認 | 証拠なし | repo 全体 |
| circuit-level | 未確認 | 証拠なし | repo 全体 |

## 3. デコードアルゴリズムの概要

### 3.1 パイプライン一覧

1. GNN 単体デコーダ
   - 実装: `panq_functions.py` / `GNNDecoder`, `logical_error_rate`
   - 利用スクリプト: `gnn_train.py`, `gnn_test.py`, `panq_nvidia.py`
2. GNN + MWPM 2段目デコーダ
   - 実装: `panq_functions.py` / `osd`
   - 利用スクリプト: `gnn_test.py`（`MatchingDecoder` を渡す設計）
3. GNN + BP/OSD 2段目デコーダ
   - 実装: `gnn_osd.py` / `logical_error_rate_osd`, `osd`
   - 利用スクリプト: `gnn_osd.py`

### 3.2 GNN 単体デコーダ

#### 明示的記述

- `GNNDecoder` は Tanner graph 上で message passing を反復する GRU ベース GNN である。
  - 根拠: `panq_functions.py` / `GNNDecoder.__init__`, `GNNDecoder.forward`
- 各反復で `msg_net` によりエッジ message を計算し、`index_add_` で受信ノードへ集約し、`GRU` でノード状態を更新する。
  - 根拠: `panq_functions.py` / `GNNDecoder.forward`
- 出力は各 iteration ごとのノード logits であり、最終的な物理エラー推定には data-qubit 部分 `error_index:` が使われる。
  - 根拠: `panq_functions.py` / `GNNDecoder.forward`, `logical_error_rate`
- 学習 loss は syndrome ノードと error ノードの cross-entropy に、`hxperp` / `hzperp` を使った `ler_loss` を加えたもの。
  - 根拠: `gnn_train.py` / top-level script
  - 根拠: `panq_functions.py` / `ler_loss`

#### 推論

- これは MWPM ではなく、Tanner graph 上の learned message-passing decoder である。
  - 根拠: `panq_functions.py` / `GNNDecoder`
- 物理 Pauli ラベル 0/1/2/3 を直接予測する multiclass decoder であり、logical observable を直接予測する方式ではない。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`, `logical_error_rate`

### 3.3 GNN + MWPM 2段目

#### 明示的記述

- `panq_functions.osd` は GNN の softmax 出力から `fllrx`, `fllrz` を計算し、`MatchingDecoder(..., weights=(fllrx[i], fllrz[i]))` を各サンプルに対して構築している。
  - 根拠: `panq_functions.py` / `osd`
- 2段目 decoding は residual syndrome が残ったサンプル `nonzero_syn_id` のみに適用される。
  - 根拠: `panq_functions.py` / `osd`

#### 推論

- このパイプラインは「GNN が qubit-wise LLR を出し、MWPM がそれを重みとして recovery を補完する」2段構成である。
  - 根拠: `panq_functions.py` / `osd`
- Y 誤りは `fllrx`, `fllrz` の 2 系統に射影され、相関付き matching ではなく独立 X/Z matching に落としている。
  - 根拠: `panq_functions.py` / `osd`

### 3.4 GNN + BP/OSD 2段目

#### 明示的記述

- `gnn_osd.py` は `BeliefPropagationOSDDecoder(code, error_model, error_rate=err_rate, osd_order=0, max_bp_iter=0)` を構築している。
  - 根拠: `gnn_osd.py` / top-level script
- `gnn_osd.py::osd` は GNN 出力から `fllrx`, `fllrz` を作り、`osd_decoder.x_decoder` と `osd_decoder.z_decoder` に別々に注入して decode している。
  - 根拠: `gnn_osd.py` / `osd`, `init_log_probs_of_decoder`

#### 推論

- ここでも BP/OSD は X/Z を独立に処理する構成であり、単一の joint Pauli graph で相関 Y を解いてはいない。
  - 根拠: `gnn_osd.py` / `osd`

### 3.5 XXZZ / XZZX などの扱い

#### 明示的記述

- repo 内の surface-code 実装は `Hx`, `Hz`, `logicals_x`, `logicals_z` を明示的に分離している。
  - 根拠: `gnn_train.py` / top-level script
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`, `logical_error_rate`, `osd`, `ler_loss`
  - 根拠: `gnn_osd.py` / `logical_error_rate_osd`, `osd`

#### 推論

- 対応しているのは CSS 型の rotated planar surface code 系であり、XXZZ 系として読むのが妥当である。
  - 根拠: `panq_functions.py` / `surface_code_edges`, `logical_error_rate`, `osd`
  - 判定: 推論
- XZZX のような mixed-basis stabilizer は repo 内で未確認であり、共通デコーダで扱う実装証拠もない。
  - 根拠: repo 全文検索結果

### 3.6 物理誤りから syndrome graph への落とし方

#### 明示的記述

- 物理誤りラベルは `0/1/2/3 = I/X/Z/Y` 相当の 4 値で持つ。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`
- 評価時は `final_targets` / `final_solution` を `final_targetsx`, `final_targetsz` と `final_solutionx`, `final_solutionz` に分解している。
  - 根拠: `panq_functions.py` / `logical_error_rate`, `osd`
  - 根拠: `gnn_osd.py` / `logical_error_rate_osd`, `osd`
- Tanner graph は stabilizer ノードと data-qubit ノードからなり、`code.stabilizer_matrix.nonzero()` に基づいて双方向エッジを張る。
  - 根拠: `panq_functions.py` / `surface_code_edges`

#### 推論

- X 誤りは Z-check 側 syndrome に、Z 誤りは X-check 側 syndrome に寄与し、Y はその両方に同時寄与する。
  - 理由: 評価・2段目処理で X/Z 成分へ分解していること、および BB 用補助実装では `syndrome_z = err_x @ hz.T`, `syndrome_x = err_z @ hx.T` と明示されている。
  - 根拠: `panq_functions.py` / `logical_error_rate`, `osd`
  - 根拠: `bb_panq_functions.py` / `generate_syndrome_error_volume`
  - 判定: 推論
- 実装は独立した X/Z matching 問題として後段デコーダに渡しており、joint decoding graph は構築していない。
  - 根拠: `panq_functions.py` / `osd`
  - 根拠: `gnn_osd.py` / `osd`

## 4. 入出力インターフェースとデコードの運用形態

### 4.1 GNN 単体パイプライン

#### 入力

- 入力データは tensor であり、`inputs` は shape `(n_nodes, n_node_inputs)` の one-hot 表現である。
  - 根拠: `panq_functions.py` / `collate`
  - 根拠: `panq_functions.py` / `adapt_trainset`
- グラフ構造自体は別 tensor の `src_ids`, `dst_ids` で与えられる。
  - 根拠: `panq_functions.py` / `collate`, `surface_code_edges`

#### Syndrome の定義

- 入力 syndrome は stabilizer 値そのものであり、前ラウンドとの差分 detection event ではない。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`
- first half / second half の stabilizer は値を `1` と `2` に分けてエンコードしている。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`
- 最終ラウンドを data-qubit readout から再構成する処理は存在しない。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`, `adapt_trainset`

#### 出力

- GNN の最終出力は各 data qubit に対する 4-class 予測であり、物理エラー列 `I/X/Z/Y` 相当を出す。
  - 根拠: `panq_functions.py` / `GNNDecoder.forward`, `logical_error_rate`

#### 運用形態

- 物理 recovery を出力して residual error を評価するため、logical readout-only decoding ではない。
  - 根拠: `panq_functions.py` / `logical_error_rate`
- 実運用コードとしては full correction / Pauli-frame 相当の recovery 推定だが、repo 内ではオフライン評価にしか使っていない。
  - 根拠: `panq_functions.py` / `logical_error_rate`
  - 根拠: `gnn_test.py` / top-level script
- active correction として回路へフィードバックする API や制御フローは未確認。
  - 根拠: repo 全体

### 4.2 GNN + MWPM パイプライン

#### 入力

- 1段目入力は GNN 単体と同じ。
  - 根拠: `gnn_test.py` / top-level script
  - 根拠: `panq_functions.py` / `collate`

#### 出力

- GNN 出力は直接の最終 recovery ではなく、`fllrx`, `fllrz` という MWPM 用 qubit-wise 重みへ変換される。
  - 根拠: `panq_functions.py` / `osd`
- 2段目 `MatchingDecoder.decode(final_syn[i])` の戻り値は、最終的に物理 recovery パターンへ戻される。
  - 根拠: `panq_functions.py` / `osd`

#### 運用形態

- full correction 用の recovery 生成であり、active feedback 実装ではない。
  - 根拠: `panq_functions.py` / `osd`, `logical_error_rate`

### 4.3 GNN + BP/OSD パイプライン

#### 入力

- 1段目入力は GNN 単体と同じ tensor + edge index。
  - 根拠: `gnn_osd.py` / top-level script
  - 根拠: `panq_functions.py` / `collate`

#### 出力

- GNN 出力は `fllrx`, `fllrz` に変換され、BP/OSD の `x_decoder`, `z_decoder` に渡される。
  - 根拠: `gnn_osd.py` / `osd`
- 2段目の出力は物理 X/Z 誤り列であり、最後に residual syndrome と logical error で評価される。
  - 根拠: `gnn_osd.py` / `osd`

#### 運用形態

- これも logical observable 直接予測ではなく、offline full correction 評価である。
  - 根拠: `gnn_osd.py` / `logical_error_rate_osd`, `osd`

## 5. Neural network 系アルゴリズムの対応

### 5.1 対応状況

- training と inference の両方に対応している。
  - 学習: `gnn_train.py`, `panq_nvidia.py`
  - 推論/評価: `gnn_test.py`, `gnn_osd.py`

### 5.2 根拠

#### 明示的記述

- README で `gnn_train.py` を training、`gnn_test.py` を decoder script と説明している。
  - 根拠: `README.md` / 本文
- `gnn_train.py` は optimizer, scheduler, backward, `save_model` を用いた学習 loop を持つ。
  - 根拠: `gnn_train.py` / top-level script
- `gnn_test.py`, `gnn_osd.py` は `load_model` で学習済み重みをロードし、エラーレート sweep を行う。
  - 根拠: `gnn_test.py` / top-level script
  - 根拠: `gnn_osd.py` / top-level script

### 5.3 合成訓練データの生成方法

#### 明示的記述

- 訓練データ生成は `generate_syndrome_error_volume` が担当する。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`
- 各サンプルについて `pr = p * rng.random()` で誤り率をサンプルし、`error_model.generate(code, pr)` で物理 Pauli 誤りを作る。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`
- その誤りから syndrome を測り、`[syndrome, error]` を結合した配列を `adapt_trainset` で one-hot 入力 + target へ変換する。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`, `adapt_trainset`
- data-qubit ノード入力はゼロ埋めであり、教師ラベル側にのみ正解物理誤りが入る。
  - 根拠: `panq_functions.py` / `adapt_trainset`

#### 推論

- 訓練は完全に合成データ駆動であり、実機データや保存済み固定 dataset は既定パスでは使っていない。
  - 理由: 既定コードでは毎回 `generate_syndrome_error_volume` を呼び、ファイルロード分岐はコメントアウトされている。
  - 根拠: `gnn_train.py` / top-level script
  - 根拠: `panq_nvidia.py` / top-level script

## 6. ベンチマークの評価内容

### 6.1 何を評価しているか

#### 明示的記述

- README には「MWPM と比較した surface code decoding」「threshold of 17%」という結果図がある。
  - 根拠: `README.md` / 本文
- `gnn_test.py` と `gnn_osd.py` は物理誤り率 `p` を sweep して `lerx`, `lerz`, `ler_tot` を記録する。
  - 根拠: `gnn_test.py` / top-level script
  - 根拠: `gnn_osd.py` / top-level script
- 評価関数 `logical_error_rate` / `logical_error_rate_osd` は、residual error `rf` に対して `code.logical_errors(rf)` と `code.measure_syndrome(rf)` を使って失敗率を数える。
  - 根拠: `panq_functions.py` / `logical_error_rate`
  - 根拠: `gnn_osd.py` / `logical_error_rate_osd`

#### 推論

- これは **logical memory 実験相当のオフライン decoding 評価** であり、1-shot readout ではない。
  - 理由: 単発の data-qubit Pauli error を syndrome から推定して residual logical/codespace error を数えているため。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`, `logical_error_rate`
  - 根拠: `gnn_osd.py` / `logical_error_rate_osd`

### 6.2 評価の前提条件

- stabilizer ラウンド数
  - 結論: 反復 syndrome round なし。単発測定のみ。
  - 根拠: `panq_functions.py` / `generate_syndrome_error_volume`
- logical Z 限定などの偏り
  - 結論: logical Z 限定評価ではない。`code.logical_errors(rf)` の任意成分が非零なら logical error と数える。
  - 根拠: `panq_functions.py` / `logical_error_rate`
  - 根拠: `gnn_osd.py` / `logical_error_rate_osd`
- しきい値図の誤りモデル
  - 結論: README は code capacity noise を明示。既定 script は `DP = PauliErrorModel(0.34, 0.32, 0.34)` を使う。
  - 根拠: `README.md` / 本文
  - 根拠: `gnn_train.py` / top-level script
  - 根拠: `gnn_test.py` / top-level script
  - 根拠: `gnn_osd.py` / top-level script

### 6.3 評価指標の注意点

#### 明示的記述

- 変数名は `lerx`, `lerz`, `ler_tot` だが、`logical_error_rate` の実装上は
  - 第1戻り値: `np.any(code.logical_errors(rf) != 0, axis=1)` に基づく logical error 率
  - 第2戻り値: residual syndrome `ms` が非零の codespace error 率
  - 第3戻り値: その OR
  - となっている。
  - 根拠: `panq_functions.py` / `logical_error_rate`
  - 根拠: `gnn_osd.py` / `logical_error_rate_osd`

#### 推論

- したがって script の表示ラベル `LER_X`, `LER_Z` は、実装内容と厳密には一致していない。
  - 根拠: `gnn_train.py` / top-level script
  - 根拠: `gnn_test.py` / top-level script
  - 根拠: `gnn_osd.py` / top-level script
  - 根拠: `panq_functions.py` / `logical_error_rate`

### 6.4 結果の存在状況

- README 図 `astra_res.png` に benchmark 結果は存在する。
  - 根拠: `README.md` / 本文
- 一方、raw 数値結果ファイルや再現用ログは repo 内に同梱されていない。`np.save(...)` はコメントアウトが多い。
  - 根拠: `gnn_test.py` / top-level script
  - 根拠: `gnn_osd.py` / top-level script
  - 根拠: `gnn_train.py` / top-level script
