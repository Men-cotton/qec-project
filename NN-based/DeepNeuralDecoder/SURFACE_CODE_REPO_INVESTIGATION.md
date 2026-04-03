# DeepNeuralDecoder Surface Code 調査メモ

このファイルは、調査完了ごとに逐次追記する作業メモ兼最終報告である。

## 1. Surface code の対応状況と実装範囲

### 1.1 対応有無

- `明示的記述`
  - この repo は surface code の QEC に対応している。根拠:
    - `README.md` の Reports 説明に `Surface` として `Rotated surface code single EC` が明記されている。根拠ファイル/関数:
      - `README.md` / 該当関数・クラスなし
    - 実行入口 `Trainer/Run.py` は `EC scheme` が `SurfaceD3` または `SurfaceD5` の場合に `_SurfaceD3Lookup` / `_SurfaceD5Lookup` を読み込む。根拠ファイル/関数:
      - `Trainer/Run.py` / main ブロック
    - surface code 用モデルクラス `Surface1EC`, `LookUpSurface1EC`, `PureErrorSurface1EC` が実装されている。根拠ファイル/関数:
      - `Trainer/ModelSurface1EC.py` / `Surface1EC`, `LookUpSurface1EC`, `PureErrorSurface1EC`

- `対象ディレクトリ`
  - `Trainer/`
    - `Trainer/ModelSurface1EC.py`
    - `Trainer/_SurfaceD3Lookup.py`
    - `Trainer/_SurfaceD5Lookup.py`
    - `Trainer/Run.py`
    - `Trainer/Networks.py`
  - `Data/Generator/Surface_1EC_D3/`
    - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m`
  - `Data/Generator/Surface_1EC_D5/`
    - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m`
  - `Data/Compact/Surface_1EC_D3/`, `Data/Compact/Surface_1EC_D5/`
    - 圧縮済みデータ形式の整形スクリプト
  - `Param/LookUp/Surface_1EC_D3`, `Param/PureError/Surface_1EC_D3`, `Param/LookUp/Surface_1EC_D5`, `Param/PureError/Surface_1EC_D5`
  - `Reports/Results/Surface_1EC_D3`, `Reports/Results/Surface_1EC_D5`

### 1.2 対応している code family / variant

- `明示的記述`
  - `README.md` は `Rotated surface code single EC` と記述している。根拠ファイル/関数:
    - `README.md` / 該当関数・クラスなし
  - Matlab 生成器は `d=3 rotated surface code`, `d=5 rotated surface code` を生成すると記述している。根拠ファイル/関数:
    - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `SurfaceCodeCircuitGenerator`, `depolarizingSimulator`
    - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `depolarizingSimulator`
  - 実装は X stabilizer と Z stabilizer を別々に持つ CSS 形式で、`err_keys=['X','Z']` と独立した `G['X']`, `G['Z']`, `L['X']`, `L['Z']` を使う。根拠ファイル/関数:
    - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
    - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`
    - `Trainer/Model.py` / `syn_from_generators`, `pure_correction`, `lookup_correction`, `check_fault_after_correction`

- `推論`
  - 実装対象は「標準的な rotated planar CSS surface code」であり、`XXZZ` や `XZZX` を切り替える仕組みは確認できない。
  - 理由:
    - stabilizer は常に X / Z に分離され、`err_keys=['X','Z']` で独立処理される。
    - generator でも `ancillaStabXMat` と `ancillaStabZMat` を分離しており、X 型・Z 型安定化子を別回路で測る。
  - 根拠ファイル/関数:
    - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
    - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`
    - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `SurfaceCodeCircuitGenerator`, `depolarizingSimulator`
    - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `SurfaceCodeCircuitGenerator`, `depolarizingSimulator`

- `未確認`
  - `XXZZ` という名称での明示的対応記述
  - `XZZX` code への対応
  - mixed-boundary / twist / heavy-hex など rotated planar 以外の surface-code variant

### 1.3 実装制約

- `明示的記述`
  - 単一 logical qubit 前提:
    - Matlab 生成器で `n=1` が固定されている。根拠ファイル/関数:
      - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `SurfaceCodeTrainingSetd3`
      - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `SurfaceCodeTrainingSetd5`
    - logical operator 行列 `L['X']`, `L['Z']` は各 1 行のみ。根拠ファイル/関数:
      - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
      - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`
  - 距離は d=3 と d=5 のみが明示実装:
    - `SurfaceD3`, `SurfaceD5` の 2 spec のみ存在する。根拠ファイル/関数:
      - `Trainer/Run.py` / main ブロック
      - `Trainer/_SurfaceD3Lookup.py` / `Spec`
      - `Trainer/_SurfaceD5Lookup.py` / `Spec`
  - 繰り返し syndrome rounds は d=3 で 3 ラウンド、d=5 で 6 ラウンド:
    - `num_syn=3` / `num_epochs=3` および `num_syn=6` / `num_epochs=6`。根拠ファイル/関数:
      - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
      - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`
    - README でも `3 rounds ... distance 3`, `6 rounds ... distance 5` と記載。根拠ファイル/関数:
      - `README.md` / 該当関数・クラスなし

- `推論`
  - odd distance 一般対応ではなく、実質的には d=3 と d=5 の固定実装である。
    - `ModelSurface1EC.choose_syndrome` は `d==3` と `d>=5` で一般 odd-distance 風の分岐を持つが、実際に供給される spec は d=3,5 だけである。根拠ファイル/関数:
      - `Trainer/ModelSurface1EC.py` / `choose_syndrome`
      - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
      - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`
  - patch shape は square patch に固定され、rectangular patch を作る API や spec は確認できない。
    - `d^2` 個の data qubit を前提に `dataMat = zeros(d,d)` を組む。根拠ファイル/関数:
      - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `SurfaceCodeCircuitGenerator`
      - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `SurfaceCodeCircuitGenerator`
  - boundaries は open boundary の planar patch とみなすのが妥当。
    - 理由は rotated surface code 表記、単一 logical qubit、四辺の境界を持つ dxd patch 生成ロジックであり、周期境界やトーリック化の処理が見当たらないため。根拠ファイル/関数:
      - `README.md` / 該当関数・クラスなし
      - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `SurfaceCodeCircuitGenerator`
      - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `SurfaceCodeCircuitGenerator`

- `未確認`
  - rectangular patch 対応
  - multi-logical patch / disjoint patch / lattice surgery patch composition
  - open boundary / rough-smooth boundary の名称による明示記述

### 1.4 未対応機能の理由整理

- `XZZX / XXZZ variant`
  - 判定: `設計上の対象外である可能性が高い`
  - 理由:
    - repo 全体が CSS code 向け general framework と README に明記されている。
    - surface code 実装も X/Z 分離の generator・correction table・logical operator を前提にしている。
  - 根拠ファイル/関数:
    - `README.md` / 該当関数・クラスなし
    - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
    - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`
    - `Trainer/Model.py` / `syn_from_generators`, `check_fault_after_correction`

- `multi-logical qubit`
  - 判定: `単に実装が見当たらない` というより `前提アーキテクチャ上の対象外`
  - 理由:
    - generator が `n=1` 固定。
    - logical operator 行列も 1 logical を前提に 1 行。
  - 根拠ファイル/関数:
    - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `SurfaceCodeTrainingSetd3`
    - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `SurfaceCodeTrainingSetd5`
    - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
    - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`

- `rectangular patch / arbitrary distance`
  - 判定: `単に実装が見当たらない`
  - 理由:
    - `SurfaceD3`, `SurfaceD5` の固定 spec しかなく、パラメトリック生成はデータ生成 Matlab 側にしかない。Python 側で d 可変の surface spec 構築は未実装。
  - 根拠ファイル/関数:
    - `Trainer/Run.py` / main ブロック
    - `Trainer/_SurfaceD3Lookup.py` / `Spec`
    - `Trainer/_SurfaceD5Lookup.py` / `Spec`

- `active correction`, `lattice surgery`
  - 判定: 現時点では `未確認`。ただし回路実行へフィードバックする API はまだ確認できていない。

### 1.5 Capability Matrix

| code family | patch shape | single/multi logical qubit | boundaries | odd distance restriction | repeated syndrome rounds | measurement error support | active correction support | lattice surgery | benchmark scripts | neural decoder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rotated planar CSS surface code, d=3 (`SurfaceD3`) | square 3x3 patch | single logical qubit | 推論: open planar boundaries | 実装は d=3 固定 | 3 rounds | あり | 未確認。少なくとも benchmark API は logical fault estimation 用 | 証拠なし | あり | あり |
| Rotated planar CSS surface code, d=5 (`SurfaceD5`) | square 5x5 patch | single logical qubit | 推論: open planar boundaries | 実装は d=5 固定 | 6 rounds | あり | 未確認。少なくとも benchmark API は logical fault estimation 用 | 証拠なし | あり | あり |

### 1.6 Capability Matrix の根拠メモ

- `code family`
  - `README.md` / 該当関数・クラスなし
  - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `SurfaceCodeCircuitGenerator`, `depolarizingSimulator`
  - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `depolarizingSimulator`
- `patch shape`
  - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `SurfaceCodeCircuitGenerator`
  - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `SurfaceCodeCircuitGenerator`
- `single/multi logical qubit`
  - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `SurfaceCodeTrainingSetd3`
  - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `SurfaceCodeTrainingSetd5`
  - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
  - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`
- `odd distance restriction`
  - `Trainer/Run.py` / main ブロック
  - `Trainer/ModelSurface1EC.py` / `choose_syndrome`
- `repeated syndrome rounds`
  - `README.md` / 該当関数・クラスなし
  - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
  - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`
- `measurement error support`
  - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `ErrorGenerator`, `depolarizingSimulator`
  - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `ErrorGenerator`, `depolarizingSimulator`
  - `Trainer/ModelSurface1EC.py` / `choose_syndrome`
- `benchmark scripts`, `neural decoder`
  - `Trainer/Run.py` / `run_benchmark`
  - `Trainer/Model.py` / `train`
  - `Trainer/Networks.py` / `ff_cost`, `rnn_cost`, `surface_conv3d_cost`, `surface_channeled_conv3d_cost`

## 2. 対象ノイズモデル

### 2.1 要約

- `明示的記述`
  - ベンチマーク / データ生成スクリプトは、surface code について Matlab 生成器 `SurfaceCodeTrainingSetd3.m`, `SurfaceCodeTrainingSetd5.m` の `ErrorGenerator` と `depolarizingSimulator` を使う。根拠ファイル/関数:
    - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `ErrorGenerator`, `depolarizingSimulator`
    - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `ErrorGenerator`, `depolarizingSimulator`
  - README は surface ベンチマークの `B Range` を `Range of physical error rates used for depolarizing noise channel` と説明している。根拠ファイル/関数:
    - `README.md` / 該当関数・クラスなし

- `推論`
  - repo 内の surface decoder 実装が直接仮定しているモデルは、「full circuit-level noise をそのまま最適化するデコーダ」ではなく、X/Z 分離された repeated-syndrome 入力から代表 syndrome を選ぶ簡約モデルである。
  - 標準ベンチマーク用のデータ生成は circuit-level depolarizing 系だが、デコーダ側は Y や CNOT 相関を独立の latent 変数としては扱わず、X/Z 成分へ射影された syndrome / error 表現を扱う。

### 2.2 デコーダ実装が仮定するノイズモデル

#### 2.2.1 Base decoder (`LookUpSurface1EC`, `PureErrorSurface1EC`)

- `明示的記述`
  - surface 用モデルは raw data から `synX`, `synZ`, `errX`, `errZ` を別々に読む。根拠ファイル/関数:
    - `Trainer/ModelSurface1EC.py` / `get_data`, `init_syn`
  - correction は `pure_correction(syn, key) = syn * T[key]` または `lookup_correction(syn, key) = correctionMat[key][index]` であり、どちらも syndrome ベクトルのみを入力に取り、`p` を使わない。根拠ファイル/関数:
    - `Trainer/Model.py` / `pure_correction`, `lookup_correction`
  - logical fault 判定も `G`, `L`, `T`, `correctionMat` に基づく X/Z 分離処理である。根拠ファイル/関数:
    - `Trainer/Model.py` / `syn_from_generators`, `lookup_correction_from_error`, `check_fault_after_correction`, `check_logical_fault`
  - repeated syndrome rounds は `choose_syndrome` で 1 つの代表 syndrome に圧縮される。根拠ファイル/関数:
    - `Trainer/ModelSurface1EC.py` / `choose_syndrome`, `abstract_init_rec`

- `推論`
  - この base decoder が仮定するのは、`measurement error を含みうる repeated syndrome history` を 1 つの代表 syndrome に要約し、その syndrome に対して X/Z 独立 correction table を当てるモデルである。
  - したがって、QEC ノイズモデル分類としては「phenomenological repeated-syndrome に近い簡約モデル」であり、full circuit-level correlation をそのまま重み付き graph で解く設計ではない。
  - 理由:
    - 時間方向情報は `choose_syndrome` のラウンド選択ロジックにしか現れず、最終 correction 自体は空間 syndrome 1 枚に対する table lookup / 線形写像である。
    - `p` に依存する重み計算や edge weight 計算は存在しない。
  - 根拠ファイル/関数:
    - `Trainer/ModelSurface1EC.py` / `choose_syndrome`, `abstract_init_rec`
    - `Trainer/Model.py` / `pure_correction`, `lookup_correction`

#### 2.2.2 measurement error の扱い

- `明示的記述`
  - d=3 では 3 ラウンドの syndrome から、等しい syndrome が 2 回出ればその syndrome を採用し、そうでなければ最後のラウンドを採用する。根拠ファイル/関数:
    - `Trainer/ModelSurface1EC.py` / `choose_syndrome`
    - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `FaultTolerantCorrectionX`, `FaultTolerantCorrectionZ`
  - d=5 では `t = floor((d-1)/2)` を使い、ラウンド間の一致・不一致回数に基づいて syndrome index を選ぶ。根拠ファイル/関数:
    - `Trainer/ModelSurface1EC.py` / `choose_syndrome`

- `推論`
  - measurement error support は `ある` が、time-like edge を持つ decoding graph で処理するのではなく、ラウンド反復と代表 syndrome 選択で吸収する実装である。

#### 2.2.3 データ誤り / Y 誤り / 相関誤り / biased noise の扱い

- `明示的記述`
  - decoder 内部表現は常に `X` と `Z` の 2 チャネルである。根拠ファイル/関数:
    - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
    - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`
    - `Trainer/ModelSurface1EC.py` / `get_data`, `init_syn`
  - network の標準 surface 実装も `predict['X']`, `predict['Z']` を返す 2 出力構造である。根拠ファイル/関数:
    - `Trainer/Networks.py` / `ff_cost`, `rnn_cost`, `surface_conv3d_cost`, `surface_channeled_conv3d_cost`

- `推論`
  - Y 誤りは独立カテゴリとしては扱わず、X/Z 成分へ分解された形でのみ扱う。
  - CNOT の相関誤りも joint object としては復号せず、結果として生じた X/Z syndrome と X/Z residual error に畳み込まれる。
  - biased noise に対する専用 weight, 専用補正表, bias parameter は decoder 実装には見当たらない。
  - 理由:
    - decoder の correction logic は `T[key]` と `correctionMat[key]` の固定表のみで、`p` や bias ratio を使わない。
    - 入出力は X/Z 2 チャネルに固定されている。
  - 根拠ファイル/関数:
    - `Trainer/Model.py` / `pure_correction`, `lookup_correction`
    - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
    - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`
    - `Trainer/Networks.py` / `ff_cost`, `rnn_cost`, `surface_conv3d_cost`, `surface_channeled_conv3d_cost`

#### 2.2.4 デコーダ仮定ノイズモデルの整理表

| 観点 | 判定 | 根拠 |
| --- | --- | --- |
| code capacity 前提か | `否` | repeated syndrome rounds を明示的に入力として使う (`num_syn`, `choose_syndrome`) |
| phenomenological 前提か | `近い` | repeated rounds を代表 syndrome に圧縮して correction table を引く |
| circuit-level correlation を直接使うか | `否` | decoder logic に回路位置や CNOT 相関の重み付けはない |
| measurement error support | `あり` | `choose_syndrome` のラウンド選択ロジック |
| データ誤りのみか | `否` | repeated rounds を使うため measurement fault も扱う |
| Y error を独立カテゴリとして扱うか | `否` | X/Z 2 チャネル固定 |
| 相関誤りを joint に扱うか | `限定的/実質的には否` | 相関 fault は X/Z 射影後の syndrome に畳み込まれるだけ |
| biased noise 前提か | `標準実装としては否` | bias 用パラメータや専用 table がない |

### 2.3 ベンチマーク / シミュレーションスクリプトが実際に使うノイズモデル

#### 2.3.1 標準 surface ベンチマーク (`Surface_1EC_D3`, `Surface_1EC_D5`)

- `明示的記述`
  - `ErrorGenerator` は circuit matrix 上で以下を注入する。
    - storage location (`Cmat(i,j)==1`): fault 発生確率 `errRate`、fault 種別 `k=randi([1,3])`
    - preparation (`Cmat==3` or `4`): fault 発生確率 `2*errRate/3`
    - measurement (`Cmat==5` or `6`): fault 発生確率 `2*errRate/3`
    - CNOT (`Cmat > 1000`): fault 発生確率 `errRate`、fault 種別 `k=randi([1,15])`
  - 根拠ファイル/関数:
    - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `ErrorGenerator`
    - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `ErrorGenerator`

- `明示的記述`
  - `PropagationStatePrepArb` は error type `1=X`, `2=Z`, `3=Y` を持ち、Y なら X/Z 両成分を立てる。CNOT では control/target の両 qubit へ誤り成分を配る。根拠ファイル/関数:
    - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `PropagationStatePrepArb`
    - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `PropagationStatePrepArb`

- `明示的記述`
  - `depolarizingSimulator` は各 EC round ごとに `ErrorGenerator(Circuit, errRate)` を呼び、round ごとの `XSyn`, `ZSyn`, `XErrTrack`, `ZErrTrack` を蓄積する。根拠ファイル/関数:
    - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `depolarizingSimulator`
    - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `depolarizingSimulator`

- `推論`
  - 標準ベンチマークの実ノイズモデルは `circuit-level stochastic Pauli noise` と整理するのが妥当である。
  - 理由:
    - ノイズ注入点が storage / prep / measurement / CNOT location ごとに定義されている。
    - CNOT では 2-qubit fault set、measurement/prep でも専用 fault がある。
    - repeated EC rounds を回路として逐次シミュレーションしている。

#### 2.3.2 measurement error / データ誤り / Y 誤り / 相関誤りの扱い

- `measurement error`
  - `明示的記述`
    - `Cmat == 5` または `6` で measurement error を `2*errRate/3` で入れる。根拠ファイル/関数:
      - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `ErrorGenerator`
      - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `ErrorGenerator`
  - 判定: `あり`

- `データ誤り`
  - `明示的記述`
    - storage location (`Cmat==1`) に fault を入れ、各ラウンドで data-qubit X/Z error string を出力する。根拠ファイル/関数:
      - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `ErrorGenerator`, `depolarizingSimulator`
      - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `ErrorGenerator`, `depolarizingSimulator`
  - 判定: `あり`

- `Y 誤り`
  - `明示的記述`
    - error type `3` は `Y` とされ、`PropagationStatePrepArb` で X/Z 両成分へ展開される。根拠ファイル/関数:
      - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `PropagationStatePrepArb`
      - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `PropagationStatePrepArb`
  - 判定: `あり`

- `相関誤り`
  - `明示的記述`
    - CNOT fault は `k=randi([1,15])` で 15 通りの 2-qubit Pauli fault を表し、`PropagationStatePrepArb` 内で control / target の両方へ配分される。根拠ファイル/関数:
      - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `ErrorGenerator`, `PropagationStatePrepArb`
      - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `ErrorGenerator`, `PropagationStatePrepArb`
  - 判定: `あり`

#### 2.3.3 code comment と実装コードのずれ

- `明示的記述`
  - `ErrorGenerator` のコメントは Hadamard / SWAP の fault model も列挙するが、実コードの分岐には `Cmat == 10` や `> 2000` に対する fault 注入処理は存在しない。根拠ファイル/関数:
    - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `ErrorGenerator`
    - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `ErrorGenerator`
  - `ErrorGenerator` のコメントには storage error の確率説明があるが、実コードでは `if xi < errRate; k = randi([1,3])` であり、実際の fault 注入はコード分岐の方に従うべきである。根拠ファイル/関数:
    - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `ErrorGenerator`
    - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `ErrorGenerator`

- `推論`
  - ベンチマークのノイズモデルを記述する際は、コメントの一般説明よりも `ErrorGenerator` の実分岐を優先して読むべきである。

#### 2.3.4 biased noise の扱い

- `明示的記述`
  - report には `Surface_1EC_D3_Biased`, `Surface_1EC_D3_B10` を参照する run が存在する。根拠ファイル/関数:
    - `Reports/Results/Surface_1EC_D3_B05/cmd.log` / 該当関数・クラスなし
    - `Reports/PureError/Surface_1EC_D3_B05/2017-12-28-14-46-56.json` / 該当関数・クラスなし
    - `Reports/LookUp/Surface_1EC_D3_B05/2017-12-28-14-44-20.json` / 該当関数・クラスなし
    - `Reports/PureError/Surface_1EC_D3_B10/2018-01-05-04-50-27.json` / 該当関数・クラスなし
    - `Reports/LookUp/Surface_1EC_D3_B10/2018-01-05-04-51-53.json` / 該当関数・クラスなし

- `未確認`
  - biased dataset の生成コード
  - bias ratio の定義
  - X/Z/Y のどれに bias が掛かっているか
  - measurement noise まで bias 対象かどうか

- `推論`
  - repo には biased surface benchmark の「結果」はあるが、「どの biased noise model で生成したか」を裏づける generator 実装は含まれていない。

### 2.4 benchmark で実際に使われたデータセットとノイズ前提の証拠

- `標準 d=3`
  - report の `env.raw folder` は `../../Data/Compact/Surface_1EC_D3/e-04/`。根拠ファイル/関数:
    - `Reports/LookUp/Surface_1EC_D3/2017-12-20-15-20-13.json` / 該当関数・クラスなし
    - `Reports/PureError/Surface_1EC_D3/2017-12-20-10-15-09.json` / 該当関数・クラスなし
  - `Data/Compact/Surface_1EC_D3/compressor.py` の header には `p`, `lu_avg`, `lu_std`, `total_size` が埋め込まれている。根拠ファイル/関数:
    - `Data/Compact/Surface_1EC_D3/compressor.py` / `run`

- `標準 d=5`
  - report の `env.raw folder` は `../../Data/Compact/Surface_1EC_D5/e-04/`。根拠ファイル/関数:
    - `Reports/LookUp/Surface_1EC_D5/2018-02-11-17-59-31.json` / 該当関数・クラスなし
    - `Reports/PureError/Surface_1EC_D5/2018-02-10-13-28-29.json` / 該当関数・クラスなし
  - `Data/Compact/Surface_1EC_D5/compressor.py` も header に `p`, `lu_avg`, `lu_std`, `total_size` を埋め込む。根拠ファイル/関数:
    - `Data/Compact/Surface_1EC_D5/compressor.py` / `run`

## 3. デコードアルゴリズムの概要

### 3.1 surface code で確認できたアルゴリズム一覧

- `明示的記述`
  - base decoder:
    - lookup table decoder (`LookUpSurface1EC`)
    - pure error decoder (`PureErrorSurface1EC`)
  - neural decoder:
    - feedforward network (`FF`)
    - recurrent neural network (`RNN`)
    - 3D CNN (`3DCNN`)
    - channel-shared 3D CNN (`Ch3DCNN`)
  - 根拠ファイル/関数:
    - `Trainer/ModelSurface1EC.py` / `LookUpSurface1EC`, `PureErrorSurface1EC`
    - `Trainer/Model.py` / `cost_function`
    - `Trainer/Networks.py` / `ff_cost`, `rnn_cost`, `surface_conv3d_cost`, `surface_channeled_conv3d_cost`
    - `Reports/Results/Surface_1EC_D3/cmd.log` / 該当関数・クラスなし
    - `Reports/Results/Surface_1EC_D5/cmd.log` / 該当関数・クラスなし
    - `Reports/PureError/Surface_1EC_D5/2018-06-09-03-54-33.json` / 該当関数・クラスなし

- `証拠なし`
  - MWPM
  - union-find
  - GNN
  - matching graph を直接解く surface decoder

### 3.2 物理誤りから syndrome への写像

- `明示的記述`
  - syndrome 計算は `syn_from_generators(err, key) = err * G[perp(key)]^T mod 2`。根拠ファイル/関数:
    - `Trainer/Model.py` / `syn_from_generators`
  - `perp('X')='Z'`, `perp('Z')='X'` なので、
    - `errX` は `G['Z']` を通して syndrome 化される
    - `errZ` は `G['X']` を通して syndrome 化される
  - 根拠ファイル/関数:
    - `Trainer/util.py` / `perp`
    - `Trainer/Model.py` / `syn_from_generators`
    - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
    - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`

- `推論`
  - surface code の X/Z チャネルは独立復号される。
  - これは「X 誤りは Z stabilizer syndrome、Z 誤りは X stabilizer syndrome」に落とす標準 CSS 的処理であり、XZZX のような交差基底 remapping や 1 つの joint syndrome graph は使っていない。

### 3.3 pipeline ごとのアルゴリズム

#### 3.3.1 Lookup pipeline

- `明示的記述`
  - `LookUpSurface1EC.init_rec` は `abstract_init_rec(..., self.lookup_correction)` を呼ぶ。根拠ファイル/関数:
    - `Trainer/ModelSurface1EC.py` / `LookUpSurface1EC.init_rec`
  - `lookup_correction` は syndrome を 2 進 index に変換し、`Spec.correctionMat[key]` の行を引く。根拠ファイル/関数:
    - `Trainer/Model.py` / `lookup_correction`
    - `Trainer/util.py` / `vec_to_index`
    - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
    - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`

- `推論`
  - これは `代表 syndrome -> 固定 correction vector` の table-based decoder であり、matching ではない。

#### 3.3.2 Pure-error pipeline

- `明示的記述`
  - `PureErrorSurface1EC.init_rec` は `abstract_init_rec(..., self.pure_correction)` を呼ぶ。根拠ファイル/関数:
    - `Trainer/ModelSurface1EC.py` / `PureErrorSurface1EC.init_rec`
  - `pure_correction` は syndrome に対して `T[key]` を掛ける線形写像である。根拠ファイル/関数:
    - `Trainer/Model.py` / `pure_correction`
    - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
    - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`

- `推論`
  - pure-error pipeline は `代表 syndrome -> 純誤り correction` の固定線形変換であり、lookup table より粗い base decoder として扱われている。

#### 3.3.3 Neural overlay pipeline

- `明示的記述`
  - `train` は `self.syn[key]` を入力に、`self.log_1hot[key]` を教師ラベルとして学習する。根拠ファイル/関数:
    - `Trainer/Model.py` / `train`
  - `self.log_1hot[key]` は `self.rec[key]` に対し `lookup_correction_from_error` を掛けた後の logical fault bit を one-hot 化したものである。根拠ファイル/関数:
    - `Trainer/Model.py` / `init_log_1hot`, `lookup_correction_from_error`, `check_fault_after_correction`
  - 推論時 `pred[key][i]` は `pred[key][i] * self.spec.L[key] + self.rec[key][i]` によって logical operator を足すかどうかの 0/1 に使われる。根拠ファイル/関数:
    - `Trainer/Model.py` / `num_logical_fault`
    - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
    - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`

- `推論`
  - NN は「物理エラー列そのもの」を直接出力していない。
  - 学習対象は `base decoder 後の residual error に対して logical operator を反転させるべきか` という binary logical label である。
  - したがって surface の NN stage は、physical correction decoder というより `logical frame / logical observable predictor` である。

#### 3.3.4 FF / RNN / 3DCNN / Ch3DCNN の違い

- `FF`
  - `明示的記述`
    - 各 `key in ['X','Z']` ごとに独立 MLP を作り、`input_size -> hidden... -> 2 labels` を計算する。根拠ファイル/関数:
      - `Trainer/Networks.py` / `ff_cost`

- `RNN`
  - `明示的記述`
    - 各 `key` ごとに `x[key]` を `[-1, num_epochs, lstm_input_size]` に reshape し、最終時刻出力で 2-class prediction を行う。根拠ファイル/関数:
      - `Trainer/Networks.py` / `rnn_cost`
      - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
      - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`

- `3DCNN`
  - `明示的記述`
    - 各 `key` ごとに syndrome history を 3D tensor に reshape し、conv3d を 2 段適用後に 2-class prediction を行う。根拠ファイル/関数:
      - `Trainer/Networks.py` / `surface_conv3d_cost`

- `Ch3DCNN`
  - `明示的記述`
    - X / Z 入力を同じ hidden feature volume に結合し、shared hidden の上に X head / Z head を別々に載せる。根拠ファイル/関数:
      - `Trainer/Networks.py` / `surface_channeled_conv3d_cost`
      - `Reports/PureError/Surface_1EC_D5/2018-06-09-03-54-33.json` / 該当関数・クラスなし

- `推論`
  - 標準 surface ベンチマークで主要に使われたのは `FF`, `RNN`, `3DCNN` であり、`Ch3DCNN` は少なくとも追加の D5 report 1 件でしか確認できない。

#### 3.3.5 joint X/Z prediction の扱い

- `明示的記述`
  - generic 実装として `mixed_ff`, `mixed_rnn`, `mixed_conv3d` は存在し、X/Z を 4-class joint label に変換して推論できる。根拠ファイル/関数:
    - `Trainer/Networks.py` / `mixed_ff`, `mixed_rnn`, `mixed_conv3d`
    - `Trainer/Model.py` / `mixed_cost_function`, `mixed_train`

- `未確認`
  - surface code の supplied param/report で `mixed_*` が使われた証拠

### 3.4 アルゴリズム整理表

| pipeline | base / neural | 実装概要 | X/Z の扱い | 使われた証拠 |
| --- | --- | --- | --- | --- |
| LookupSurface1EC | base | 代表 syndrome を correction table に引く | 独立 | `ModelSurface1EC`, `Model.lookup_correction` |
| PureErrorSurface1EC | base | 代表 syndrome に pure-error 行列 `T[key]` を掛ける | 独立 | `ModelSurface1EC`, `Model.pure_correction` |
| FF | neural | syndrome 履歴ベクトルから 2-class MLP | 独立 | D3/D5 reports, `ff_cost` |
| RNN | neural | syndrome 履歴を sequence として LSTM/GRU へ入力 | 独立 | D3/D5 reports, `rnn_cost` |
| 3DCNN | neural | syndrome 履歴を 3D tensor とみなし conv3d | 独立 | D5 reports, `surface_conv3d_cost` |
| Ch3DCNN | neural | X/Z を共有 feature volume に結合し別 head 出力 | 入力共有・出力は分離 | D5 PE report 1 件, `surface_channeled_conv3d_cost` |
| MixedFF / MixedRNN / MixedConv3d | neural | 4-class joint X/Z prediction | joint label | 実装あり、surface 実行証拠は未確認 |

## 4. 入出力インターフェースとデコードの運用形態

### 4.1 共通入力フォーマット

- `明示的記述`
  - `get_data` は 1 行を分解して `synX`, `synZ`, `errX`, `errZ` を matrix 化する。根拠ファイル/関数:
    - `Trainer/ModelSurface1EC.py` / `get_data`
  - D3 compact data のコメント上の format は
    - `synX1 synX2 synX3 errX1 errX2 errX3 synZ1 synZ2 synZ3 errZ1 errZ2 errZ3`
  - D5 compact data も同様に各ラウンドの `syn` / `err` を連結して出力する。根拠ファイル/関数:
    - `Data/Compact/Surface_1EC_D3/compressor.py` / `run`
    - `Data/Compact/Surface_1EC_D5/compressor.py` / `run`

- `推論`
  - input syndrome は detection event の差分ではなく、各ラウンドの stabilizer measurement result をそのまま連結したものと読むべきである。
  - 理由:
    - compressor / parser のどこにも round 間 XOR や difference を取る処理がない。
    - `choose_syndrome` も raw round syndrome の一致/不一致だけを見ている。
  - 根拠ファイル/関数:
    - `Trainer/ModelSurface1EC.py` / `get_data`, `choose_syndrome`
    - `Data/Compact/Surface_1EC_D3/compressor.py` / `run`
    - `Data/Compact/Surface_1EC_D5/compressor.py` / `run`

- `未確認`
  - syndrome bit が `+1/-1` stabilizer 値から変換されたものか、直接 `0/1` defect bit として書き出されたものかの定義文

### 4.2 syndrome の最終ラウンド再構成の有無

- `明示的記述`
  - `depolarizingSimulator` は各ラウンドで `XSyn`, `ZSyn` を直接出力し、parser もそれをそのまま読む。根拠ファイル/関数:
    - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `depolarizingSimulator`
    - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `depolarizingSimulator`
    - `Trainer/ModelSurface1EC.py` / `get_data`

- `推論`
  - 最終 round syndrome を data-qubit readout から再構成している証拠はない。
  - 入力は ancilla 測定由来の syndrome history であり、最終データ readout は `errX`, `errZ` 側にだけ現れる。

### 4.3 パイプラインごとの入力データ型

#### 4.3.1 Base decoder 共通

- `明示的記述`
  - `self.syn['X']`, `self.syn['Z']` は `np.matrix` の binary tensor で、shape は `[N, input_size]`。根拠ファイル/関数:
    - `Trainer/ModelSurface1EC.py` / `get_data`, `init_syn`
    - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
    - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`

#### 4.3.2 FF

- `明示的記述`
  - TensorFlow placeholder は `x[key]: tf.float32 [None, spec.input_size]`。根拠ファイル/関数:
    - `Trainer/Model.py` / `train`
    - `Trainer/Networks.py` / `ff_cost`

#### 4.3.3 RNN

- `明示的記述`
  - `x[key]` は flat vector で placeholder された後、`[-1, spec.num_epochs, spec.lstm_input_size]` に reshape される。根拠ファイル/関数:
    - `Trainer/Model.py` / `train`
    - `Trainer/Networks.py` / `rnn_cost`
    - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
    - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`

#### 4.3.4 3DCNN / Ch3DCNN

- `明示的記述`
  - `3DCNN`:
    - `x[key]` を `[-1, spec.num_syn, spec.syn_w[key], spec.syn_h[key], 1]` に reshape する。根拠ファイル/関数:
      - `Trainer/Networks.py` / `surface_conv3d_cost`
  - `Ch3DCNN`:
    - `x['X']` と `x['Z']` を別 reshape 後に channel 方向で concat する。根拠ファイル/関数:
      - `Trainer/Networks.py` / `surface_channeled_conv3d_cost`

### 4.4 パイプラインごとの出力データ

#### 4.4.1 Base decoder 内部出力

- `明示的記述`
  - `abstract_init_rec` は `self.rec[key] = raw_data['err'+key] + abs_corr(rep_syn[key], key)` を作る。根拠ファイル/関数:
    - `Trainer/ModelSurface1EC.py` / `abstract_init_rec`

- `推論`
  - base decoder の内部出力は「代表 syndrome に基づく data-qubit correction を適用した residual physical error string」である。
  - ただしこの residual は外部 API で返されるのではなく、後段 NN の教師ラベル生成と評価の中間状態として使われる。

#### 4.4.2 Neural pipeline 出力

- `明示的記述`
  - `predict[key] = tf.argmax(logits[key], 1)` により各 key ごとに 0/1 の class を返す。根拠ファイル/関数:
    - `Trainer/Networks.py` / `ff_cost`, `rnn_cost`, `surface_conv3d_cost`, `surface_channeled_conv3d_cost`
  - `num_logical_fault` はこの 0/1 を `pred[key][i] * self.spec.L[key]` として logical operator の適用有無に解釈する。根拠ファイル/関数:
    - `Trainer/Model.py` / `num_logical_fault`
    - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
    - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`

- `推論`
  - NN の正確な出力は `物理エラー string` でも `MWPM edge weight` でもなく、`論理オブザーバブル / logical frame の直接予測 bit` である。

### 4.5 訓練ラベルの定義

- `明示的記述`
  - `init_log_1hot` は
    - `err = check_fault_after_correction((rec + lookup_correction_from_error(rec)) % 2, key)`
    - `log_1hot[key] = y2indicator(err, 2)`
    としてラベルを作る。根拠ファイル/関数:
    - `Trainer/Model.py` / `init_log_1hot`

- `推論`
  - 教師ラベルは `residual error に追加の logical correction が必要か` を表す 2-class label であり、exact decoder output string ではない。
  - pure-error mode でも label 作成と最終評価に `lookup_correction_from_error` が使われるため、pure-error base の上で logical fault class を学習する形になっている。

### 4.6 運用形態の判定

- `full correction`
  - `限定的`
  - 理由:
    - base decoder は内部で data-qubit correction vector を生成する。
    - しかし NN 出力は physical correction string ではない。
  - 根拠ファイル/関数:
    - `Trainer/ModelSurface1EC.py` / `abstract_init_rec`
    - `Trainer/Model.py` / `num_logical_fault`

- `Pauli frame update`
  - `該当`
  - 理由:
    - 推論結果は `pred * L[key]` として logical operator を加える 0/1 bit に使われる。
  - 根拠ファイル/関数:
    - `Trainer/Model.py` / `num_logical_fault`

- `logical readout-only decoding`
  - `該当`
  - 理由:
    - benchmark の最終評価値は `num_logical_fault` による logical failure rate であり、NN 自身は logical fault bit を予測する。
  - 根拠ファイル/関数:
    - `Trainer/Model.py` / `num_logical_fault`
    - `Trainer/Run.py` / `run_benchmark`

- `active correction として回路へフィードバック可能か`
  - `未確認。少なくとも repo 内に回路制御 API はない`
  - 根拠:
    - benchmark driver は offline 学習と logical fault rate 評価のみを行う。根拠ファイル/関数:
      - `Trainer/Run.py` / `run_pickler`, `run_benchmark`

### 4.7 I/O と運用形態の整理表

| pipeline | 入力 | syndrome 定義 | 出力 | 運用形態 |
| --- | --- | --- | --- | --- |
| LookupSurface1EC | `np.matrix` binary syndrome history | raw stabilizer round results を連結 | 内部 residual physical error string (`rec`) | offline base correction |
| PureErrorSurface1EC | 同上 | 同上 | 内部 residual physical error string (`rec`) | offline base correction |
| FF | flat tensor `[None, input_size]` | 同上 | X/Z ごとの 0/1 logical bit | Pauli frame / logical readout |
| RNN | flat tensor を sequence reshape | 同上 | X/Z ごとの 0/1 logical bit | Pauli frame / logical readout |
| 3DCNN | flat tensor を 3D reshape | 同上 | X/Z ごとの 0/1 logical bit | Pauli frame / logical readout |
| Ch3DCNN | X/Z 2 tensor を channel merge | 同上 | X/Z ごとの 0/1 logical bit | Pauli frame / logical readout |

## 5. Neural network 系アルゴリズムの対応

### 5.1 training / inference 対応状況

- `明示的記述`
  - training:
    - `Run.py bench` は pickled model を読み、`m.train(param)` ないし `m.iso_train(param)`, `m.mixed_train(param)` を実行する。根拠ファイル/関数:
      - `Trainer/Run.py` / `run_benchmark`
      - `Trainer/Model.py` / `train`, `iso_train`, `mixed_train`
  - inference:
    - `train` は学習後に `predict[key]` を test set に対して計算して返す。根拠ファイル/関数:
      - `Trainer/Model.py` / `train`
  - supplied surface reports でも FF / RNN / 3DCNN / Ch3DCNN の結果が存在する。根拠ファイル/関数:
    - `Reports/Results/Surface_1EC_D3/cmd.log` / 該当関数・クラスなし
    - `Reports/Results/Surface_1EC_D5/cmd.log` / 該当関数・クラスなし
    - `Reports/PureError/Surface_1EC_D5/2018-06-09-03-54-33.json` / 該当関数・クラスなし

- `結論`
  - NN 系は `training` と `inference` の両方に対応している。

### 5.2 合成訓練データの生成フロー

- `明示的記述`
  - README は Matlab による circuit simulation が必要と述べている。根拠ファイル/関数:
    - `README.md` / 該当関数・クラスなし
  - D3 generator は Matlab で `SyndromeAndError*.txt` を書く。根拠ファイル/関数:
    - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `SurfaceCodeTrainingSetd3`
  - D5 generator は Matlab で `SyndromeOnly*.txt` と `ErrorOnly*.txt` を別々に書く。根拠ファイル/関数:
    - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `SurfaceCodeTrainingSetd5`
  - compressor がこれらを compact text format に整形する。根拠ファイル/関数:
    - `Data/Compact/Surface_1EC_D3/compressor.py` / `run`
    - `Data/Compact/Surface_1EC_D5/compressor.py` / `run`
  - `Run.py gen` は compact text を読んで `LookUpSurface1EC` または `PureErrorSurface1EC` を pickle 化する。根拠ファイル/関数:
    - `Trainer/Run.py` / `run_pickler`
  - `Run.py bench` はその pickle を用いて学習・評価する。根拠ファイル/関数:
    - `Trainer/Run.py` / `run_benchmark`

- `推論`
  - surface 用 synthetic training data の実フローは
    - circuit-level Matlab simulator
    - compact text conversion
    - Python model/pickle 化
    - TensorFlow training
    である。

### 5.3 synthetic data に含まれる教師情報

- `明示的記述`
  - raw compact data は syndrome history と最終 round の `errX`, `errZ` を含む。根拠ファイル/関数:
    - `Trainer/ModelSurface1EC.py` / `get_data`
    - `Data/Compact/Surface_1EC_D3/compressor.py` / `run`
    - `Data/Compact/Surface_1EC_D5/compressor.py` / `run`
  - pickle 化の際に base decoder が `rec` を作り、`init_log_1hot` で logical label を生成する。根拠ファイル/関数:
    - `Trainer/Model.py` / `init_log_1hot`
    - `Trainer/ModelSurface1EC.py` / `abstract_init_rec`

- `推論`
  - NN の教師ラベルは Matlab generator が直接出力するものではなく、
    - synthetic syndrome / error data
    - base decoder の residual 化
    - lookup-based logical fault 判定
    を経て Python 側で二値ラベル化されたものである。

### 5.4 未確認事項

- `未確認`
  - 実機データや外部データセットを surface NN に流し込む supplied pipeline
  - biased surface dataset の生成コード

## 6. ベンチマークの評価内容

### 6.1 何を評価しているか

- `明示的記述`
  - plot script の縦軸は `Logical fault rate`、横軸は `Physical fault rate`。根拠ファイル/関数:
    - `Reports/poly_plot.py` / `plot_results`
    - `Reports/simple_plot.py` / `plot_results`
  - `run_benchmark` は `m.num_logical_fault(prediction, test_beg)` を評価値として記録する。根拠ファイル/関数:
    - `Trainer/Run.py` / `run_benchmark`
    - `Trainer/Model.py` / `num_logical_fault`

- `結論`
  - 既存 benchmark は `logical memory / repeated EC に対する logical fault rate` を評価している。
  - `1-shot readout` を直接評価している証拠はない。

### 6.2 評価対象の偏り

- `明示的記述`
  - `num_logical_fault` は `for key in self.spec.err_keys` で X/Z 両方を調べ、どちらか一方でも logical fault があればその sample を failure として数える。根拠ファイル/関数:
    - `Trainer/Model.py` / `num_logical_fault`

- `推論`
  - report の logical fault rate は `logical X` と `logical Z` を分離した指標ではなく、`X または Z のどちらかで失敗したら 1` という合算 failure rate である。

### 6.3 評価の前提条件

- `round 数`
  - `明示的記述`
    - d=3: 3 rounds。根拠ファイル/関数:
      - `README.md` / 該当関数・クラスなし
      - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
      - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `depolarizingSimulator`
    - d=5: 6 rounds。根拠ファイル/関数:
      - `README.md` / 該当関数・クラスなし
      - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`
      - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `depolarizingSimulator`

- `評価対象`
  - `明示的記述`
    - 単一 logical qubit。根拠ファイル/関数:
      - `Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
      - `Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`
      - `Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `SurfaceCodeTrainingSetd3`
      - `Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `SurfaceCodeTrainingSetd5`

- `ベースライン比較`
  - `明示的記述`
    - report JSON には `lu avg`, `lu std` が記録され、plot script も lookup table baseline を常に描く。根拠ファイル/関数:
      - `Trainer/Run.py` / `run_benchmark`
      - `Reports/poly_plot.py` / `plot_results`
      - `Reports/simple_plot.py` / `plot_results`

- `物理誤り率範囲`
  - `明示的記述`
    - README の surface benchmark range:
      - d=3: `1e-4` から `6e-4`
      - d=5: `3e-4` から `8e-4`
    - 根拠ファイル/関数:
      - `README.md` / 該当関数・クラスなし

- `しきい値図かどうか`
  - `推論`
    - supplied plot は threshold extraction そのものではなく、physical fault rate に対する logical fault rate の log-log curve と polynomial fit である。
  - 根拠ファイル/関数:
    - `Reports/poly_plot.py` / `plot_results`
    - `Reports/simple_plot.py` / `plot_results`

### 6.4 D3 / D5 で提供されている benchmark の実体

- `d=3`
  - `明示的記述`
    - README 表では surface d=3 に対して PU/LU の双方で `RNN`, `FF0`, `FF1`, `FF2` の結果がある。根拠ファイル/関数:
      - `README.md` / 該当関数・クラスなし
    - `Reports/Results/Surface_1EC_D3/cmd.log` でも LU-FF0/1/2, LU-RNN, PE-FF1/2, PE-RNN を比較している。根拠ファイル/関数:
      - `Reports/Results/Surface_1EC_D3/cmd.log` / 該当関数・クラスなし

- `d=5`
  - `明示的記述`
    - README 表では surface d=5 に対して PU/LU の双方で `RNN`, `FF2`, `CNN` の結果がある。根拠ファイル/関数:
      - `README.md` / 該当関数・クラスなし
    - `Reports/Results/Surface_1EC_D5/cmd.log` でも PE-FF2, PE-RNN, PE-3D-CNN と LU-FF2, LU-RNN, LU-3D-CNN を比較している。根拠ファイル/関数:
      - `Reports/Results/Surface_1EC_D5/cmd.log` / 該当関数・クラスなし

### 6.5 benchmark 解釈時の注意

- `明示的記述`
  - `data['fault scale'] = m.error_scale = data_size / total_size` が掛けられ、最終結果 `Result` は `m.error_scale * result` として記録される。根拠ファイル/関数:
    - `Trainer/Model.py` / `__init__`
    - `Trainer/Run.py` / `run_benchmark`

- `推論`
  - compact dataset では all-zero sample を落としているため、report の logical fault rate は raw Monte Carlo 全体に対する rate へ `fault scale` で再スケーリングされた値である。
  - 根拠ファイル/関数:
    - `Data/Compact/Surface_1EC_D3/compressor.py` / `run`
    - `Data/Compact/Surface_1EC_D5/compressor.py` / `run`
    - `Trainer/Model.py` / `__init__`
    - `Trainer/Run.py` / `run_benchmark`
