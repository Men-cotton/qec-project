# Repo Investigation: Surface Code Decoder

最終更新: 2026-03-20

この文書は、各調査段階の終了時点で追記する。主張ごとに、根拠となるファイルパスと関数/クラス名を明記し、`明示的記述` と `推論` を区別する。証拠がない項目は `未確認` と記す。

## 1. Surface code の対応状況と実装範囲

### 結論

- この repo は surface code の QEC に対応している。
  - 明示的記述: README が「Graph neural network decoder for the rotated surface code」と述べている。[根拠: `README.md`, 記載本文, lines 1-2]
  - 明示的記述: 実装は `stim.Circuit.generated("surface_code:rotated_memory_z", ...)` を直接生成している。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 135-161]

- 調査対象ディレクトリは `src/`, ルートの設定 YAML 群, `models/` である。
  - 明示的記述: README が `src/` をソースコード、`models/` を学習済みモデル置き場として説明している。[根拠: `README.md`, 記載本文, lines 45-61]
  - 明示的記述: ルートに `config_surface_codes_3_3.yaml` と `config_surface_codes_9_3.yaml` があり、`train_nn.py` がそれらの設定を `Decoder` に渡す。[根拠: `train_nn.py`, `main`, lines 11-22]

### 対応しているコード種別

- 明示的記述:
  - 実装で明示されているのは `surface_code:rotated_memory_z` のみであり、README でも `rotated surface code` としか述べていない。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 135-161; `README.md`, 記載本文, lines 1-2]
  - syndrome 表現は `1 = violated X-stabilizer`, `3 = violated Z-stabilizer` として符号化される。[根拠: `src/decoder.py`, `Decoder.stim_to_syndrome_3D`, lines 181-194]

- 推論:
  - 現行ソースが前提としているのは、X 型 stabilizer と Z 型 stabilizer を分けて持つ CSS 系の rotated surface code であり、少なくともソース上に XZZX 固有の切替や分岐は存在しない。[根拠: `src/decoder.py`, `Decoder.stim_to_syndrome_3D`, lines 181-194; `src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 196-229]
  - `XXZZ` という語そのものは repo 内に現れず、`XZZX` も現れないため、バリアント名として明示確認できるのは `rotated_memory_z` だけである。XZZX 対応は未確認ではなく、現行ソースからは支持できない。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 135-161; `README.md`, 記載本文, lines 1-2]

### 実装上の制約

- 単一 logical qubit 出力に制約されている。
  - 明示的記述: 既存設定ファイルは `num_classes: 1` を指定している。[根拠: `config_surface_codes_3_3.yaml`, 記載本文, lines 5-18; `config_surface_codes_9_3.yaml`, 記載本文, lines 5-21]
  - 明示的記述: `GNN_7` の出力次元は `num_classes` で決まり、現在の訓練・評価ではその出力を `observable_flips` と比較している。[根拠: `src/gnn_models.py`, `GNN_7.__init__`, lines 15-42; `src/decoder.py`, `Decoder.train`, lines 297-313; `src/decoder.py`, `Decoder.evaluate_test_set`, lines 231-245]
  - 推論: 現行パイプラインは 1 個の logical observable の二値分類を行う構成であり、multi-logical decoding 用のインターフェースは実装されていない。[根拠: `src/gnn_models.py`, `GNN_7.__init__`, lines 15-42; `src/graph_representation.py`, `sample_syndromes`, lines 33-49]

- repeated syndrome rounds には対応している。
  - 明示的記述: `rounds=self.d_t` で Stim circuit を生成する。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 137-158]
  - 明示的記述: syndrome mask の時間軸長は `self.d_t + 1` であり、ノード位置 `pos` に時間座標が含まれる。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 173-179; `src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 218-227]

- measurement error を含む時系列 detector event を扱う。
  - 明示的記述: circuit 生成時に `before_measure_flip_probability` と `after_reset_flip_probability` を設定している。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 137-158]
  - 明示的記述: 入力は stabilizer 値そのものではなく `detection_events` であり、時間座標付き 3D syndrome へ変換される。[根拠: `src/graph_representation.py`, `sample_syndromes`, lines 15-49; `src/decoder.py`, `Decoder.stim_to_syndrome_3D`, lines 181-194]

- active correction / lattice surgery は現行ソースの設計対象外に見える。
  - 明示的記述: 予測対象は `observable_flips` であり、物理 qubit への補正列や回路へのフィードバックを返す API は存在しない。[根拠: `src/graph_representation.py`, `sample_syndromes`, lines 15-49; `src/decoder.py`, `Decoder.evaluate_test_set`, lines 231-245; `src/decoder.py`, `Decoder.test`, lines 370-402]
  - 推論: `surface_code:rotated_memory_z` を固定生成する単一メモリ実験パイプラインであるため、lattice surgery のような複数パッチ/操作列を扱う設計ではない。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 135-161]

- odd distance 限定かどうかは未確認。
  - 明示的記述: 既存設定ファイルと保存済みモデルのファイル名は `d=3,5,7,9,11,13,15` の奇数距離のみを示している。[根拠: `config_surface_codes_3_3.yaml`, 記載本文, lines 19-25; `config_surface_codes_9_3.yaml`, 記載本文, lines 22-28; `models/circuit_level_noise/` 配下ファイル名一覧]
  - 推論: artifact 群は odd distance のみだが、`code_size` に対する明示的なバリデーションは実装されていないため、ソースレベルで odd-only 制約が強制されているとは断定できない。[根拠: `src/decoder.py`, `Decoder.__init__`, lines 33-38; `src/decoder.py`, `Decoder.initialise_simulations`, lines 135-179]

- patch shape, boundaries, rectangular patch 可否は repo 単体では未確認。
  - 明示的記述: `distance=self.code_size` という単一スカラーだけが渡されており、縦横別距離や boundary 種別を設定する引数はこの repo には存在しない。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 137-158]
  - 推論: 現行インターフェースでは rectangular patch や boundary 切替を露出していない。これは「未実装」の可能性と「固定 generator を使う設計」の両方があるが、repo 内だけではどちらかを確定できない。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 135-161]

### Capability Matrix

| code family | patch shape | single/multi logical qubit | boundaries | odd distance restriction | repeated syndrome rounds | measurement error support | active correction support | lattice surgery | benchmark scripts | neural decoder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rotated surface code (`surface_code:rotated_memory_z`) | 未確認。単一 `distance` 指定のみが露出 | single のみ確認。`num_classes=1` | 未確認 | 未確認。artifact は odd のみ | 対応 (`rounds=self.d_t`) | 対応 (`before_measure_flip_probability`) | 非対応 | 非対応 | あり (`train_nn.py`, `Decoder.train`, `Decoder.test`) | あり (`GNN_7`) |

### 未対応機能の整理

- XZZX variant:
  - 判定: 現行ソースでは未対応とみなすのが妥当。
  - 理由: generator 名が固定で `surface_code:rotated_memory_z`、かつ syndrome 表現も X stabilizer / Z stabilizer の二種を前提にしており、XZZX 切替ロジックがない。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 135-161; `src/decoder.py`, `Decoder.stim_to_syndrome_3D`, lines 181-194]
  - 区別: 「単に実装が見当たらない」というより、現行アーキテクチャが固定 generator に依存しており対象外になっている可能性が高い。ただし repo 内の明示説明はないため、最終分類は推論。

- multi-logical qubit:
  - 判定: 未対応。
  - 理由: モデル出力と設定が 1 ビットの logical observable 予測に固定されている。[根拠: `config_surface_codes_3_3.yaml`, 記載本文, lines 5-18; `src/gnn_models.py`, `GNN_7.__init__`, lines 15-42]
  - 区別: これは現行実装上の未対応であり、単なる未確認ではない。

- rectangular patch / boundary 選択:
  - 判定: 未確認。
  - 理由: 露出 API がないことは確認できるが、基盤の Stim generator が内部的に何を許すかは、この repo 単体では検証できていない。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 135-161]

## 2. 対象ノイズモデル

### 2.1 デコーダ実装が仮定するノイズモデル

- 明示的記述:
  - 現行デコーダは `stim.Circuit.generated("surface_code:rotated_memory_z", ...)` から得た `detection_events` を入力にする。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 135-161; `src/graph_representation.py`, `sample_syndromes`, lines 15-49]
  - 入力グラフ化は detector 座標と `X/Z` ラベルだけを使っており、MWPM のようなノイズ依存の edge weight 計算は実装していない。エッジ属性は `1 / dist ** self.power` の幾何学的重みである。[根拠: `src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 196-229]

- 推論:
  - デコーダロジック自体は analytic な code-capacity / phenomenological / circuit-level 専用重み付けを持たず、学習データ分布に依存する data-driven decoder である。
  - ただし、repo に同梱された実行可能パイプラインが生成するデータは `surface_code:rotated_memory_z` の circuit-level memory 実験なので、実用上はそのノイズモデルを前提にした decoder とみなすのが妥当である。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 135-161; `src/decoder.py`, `Decoder.train`, lines 264-283]

- measurement error の扱い:
  - 明示的記述: `before_measure_flip_probability` と時間軸 `d_t + 1` を使うため、measurement error を含む detector event 列を扱う。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 137-179]

- データ誤りのみか:
  - 明示的記述: `before_round_data_depolarization` を使うため、データ qubit 側の誤りも入る。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 142-145, 155-158]

- 相関誤り / Y 誤り:
  - 明示的記述: repo が直接制御しているのは Stim への depolarization / flip probability の引数指定のみで、X/Y/Z の内訳や相関構造を repo 側で分解していない。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 137-158]
  - 推論: Y 誤りや相関誤りが存在するとしても、それらは Stim 側で detector event に畳み込まれ、repo 内では個別ラベルを持たない。

- biased noise:
  - 明示的記述: すべてのノイズ注入箇所に同じ `error_rate` または `p` を渡している。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 137-158]
  - 推論: bias を表現する独立パラメータは現行ソースに存在しないため、biased-noise 前提の decoder ではない。

### 2.2 ベンチマーク / シミュレーションスクリプトが実際に使うノイズモデル

- 現行 checkout で実行可能なベンチマークスクリプトは circuit-level noise を使う。
  - 明示的記述: `train_nn.py` は `Decoder.run()` を呼び、`Decoder.train` / `Decoder.test` は `initialise_simulations` で生成した Stim sampler からデータを作る。[根拠: `train_nn.py`, `main`, lines 11-22; `src/decoder.py`, `Decoder.run`, lines 405-418; `src/decoder.py`, `Decoder.train`, lines 264-283; `src/decoder.py`, `Decoder.test`, lines 375-391]
  - 明示的記述: その simulator は `after_clifford_depolarization`, `after_reset_flip_probability`, `before_measure_flip_probability`, `before_round_data_depolarization` を同時に有効化している。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 137-158]

- 訓練時の具体条件:
  - 明示的記述: `config_surface_codes_3_3.yaml` と `config_surface_codes_9_3.yaml` は `train_error_rate: [0.001, ..., 0.005]`, `test_error_rate: 0.001`, `d_t: 3` を指定する。[根拠: `config_surface_codes_3_3.yaml`, 記載本文, lines 19-25; `config_surface_codes_9_3.yaml`, 記載本文, lines 22-28]
  - 明示的記述: train 側では error rate の list を sampler 群にして interleave し、test 側では単一 `test_error_rate` を使う。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 148-161; `src/graph_representation.py`, `sample_syndromes`, lines 6-32]

- `perfect_stabilizers` / `repetition_code` artifact 群について:
  - 明示的記述: README はそれらの学習済みモデルが `models/` にあると書いている。[根拠: `README.md`, 記載本文, lines 58-59]
  - 明示的記述: しかし現行ソースの実行スクリプトと `Decoder` 実装は `surface_code:rotated_memory_z` 以外の generator 分岐を持たない。[根拠: `train_nn.py`, `main`, lines 11-22; `src/decoder.py`, `Decoder.initialise_simulations`, lines 135-161]
  - 判定: これら artifact の「現在の checkout で再生成されるノイズモデル」は未確認。少なくとも、対応する script / class は現行ソースからは確認できない。

## 3. デコードアルゴリズムの概要

### 3.1 現行ソースで確認できるパイプライン

- GNN decoder パイプラインのみが明示実装されている。
  - 明示的記述: README は GNN decoder を説明し、`src/gnn_models.py` に PyTorch Geometric モデルがあると述べる。[根拠: `README.md`, 記載本文, lines 45-49]
  - 明示的記述: `Decoder` は `GNN_7` を instantiate し、`GraphConv` 層群と MLP で予測する。[根拠: `src/decoder.py`, `Decoder.__init__`, lines 65-72; `src/gnn_models.py`, `GNN_7`, lines 6-61]

- MWPM / CNN は現行ソースには存在しない。
  - 明示的記述: `src/` とルート script は `Decoder`, `GNN_7`, graph 生成、設定読込のみで構成される。[根拠: `README.md`, 記載本文, lines 45-58; `train_nn.py`, `main`, lines 11-22]
  - 判定: MWPM / CNN の実装は未確認ではなく、現行 checkout の source code には見当たらない。

### 3.2 処理フロー

1. Stim memory circuit を生成し detector sampler を作る。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 135-161]
2. `sample_syndromes` が `detection_events` と `observable_flips` をサンプルし、空 syndrome を除外する。[根拠: `src/graph_representation.py`, `sample_syndromes`, lines 4-49]
3. `stim_to_syndrome_3D` が detector event を 3D syndrome 格子へ配置し、X stabilizer を `1`、Z stabilizer を `3` と符号化する。[根拠: `src/decoder.py`, `Decoder.stim_to_syndrome_3D`, lines 181-194]
4. `get_batch_of_graphs` が defect ごとにノードを作り、特徴量 `[is_X, is_Z, x, z, t]` と kNN グラフ、距離ベース edge attribute を構成する。[根拠: `src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 196-229]
5. `GNN_7` が graph embedding を作り、最終的に logical observable flip の logits を出す。[根拠: `src/gnn_models.py`, `GNN_7.forward`, lines 44-61]
6. 学習・評価は `BCEWithLogitsLoss` で行い、しきい値 `0.5` の二値化で正解率を計算する。[根拠: `src/decoder.py`, `Decoder.train`, lines 257-321; `src/decoder.py`, `Decoder.evaluate_test_set`, lines 231-245]

### 3.3 X/Z variant の扱い

- 明示的記述:
  - X stabilizer と Z stabilizer は別ノード種別として区別される。[根拠: `src/decoder.py`, `Decoder.stim_to_syndrome_3D`, lines 181-194; `src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 204-212]
  - ただしグラフ自体は一本で、`knn_graph(pos, ...)` は node type ごとに分割せず空間時間座標だけで辺を張る。[根拠: `src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 218-227]

- 推論:
  - この実装は「独立した X matching と Z matching」を行う MWPM 型ではなく、X/Z defect を同一グラフ上で joint に処理する GNN decoder である。
  - 物理 X/Z/Y 誤りがどの detector event を起こすかという細部は repo 内で手計算しておらず、Stim circuit generator と detector sampler に委譲されている。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 135-161; `src/graph_representation.py`, `sample_syndromes`, lines 15-49]

### 3.4 旧 artifact 群について

- 明示的記述:
  - `models/perfect_stabilizers/*.pt` と `models/repetition_code/*.pt` は存在する。[根拠: `models/perfect_stabilizers/`, `models/repetition_code/`, ファイル一覧]
  - README もそれらを別 figure に対応する学習済みモデルとして言及する。[根拠: `README.md`, 記載本文, lines 58-59]

- 判定:
  - これら artifact に対応する algorithm pipeline の source code は現行 checkout では未確認。少なくとも `src/decoder.py` と `train_nn.py` から到達可能な実装は、rotated surface code 用 GNN pipeline だけである。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 135-161; `train_nn.py`, `main`, lines 11-22]

## 4. 入出力インターフェースとデコードの運用形態

### 4.1 入力データ

- raw simulator 出力:
  - 明示的記述: `sample_syndromes` の返り値は `detection_events` の NumPy 配列、`observable_flips` の Torch tensor、`n_trivial_syndromes` である。[根拠: `src/graph_representation.py`, `sample_syndromes`, lines 45-49]

- GNN への実入力:
  - 明示的記述: `get_batch_of_graphs` は `x`, `edge_index`, `batch_labels`, `edge_attr` を返す。[根拠: `src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 196-229]
  - 明示的記述: `x` は float tensor、`batch_labels` は long tensor、`edge_attr` は距離ベースの float tensor である。[根拠: `src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 207-229]
  - 推論: ノード特徴量は `[is_X, is_Z, x, z, t]` の 5 次元である。`node_features` 6 列のうち batch index 列を除いた `x_cols = [0,1,3,4,5]` がモデル入力になるため。[根拠: `src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 207-219]

### 4.2 Syndrome の定義

- 明示的記述:
  - repo が直接扱うのは stabilizer 生値ではなく `detection_events` である。[根拠: `src/graph_representation.py`, `sample_syndromes`, lines 15-49; `src/decoder.py`, `Decoder.stim_to_syndrome_3D`, lines 181-194]
  - repo 内に前ラウンドとの差分を自前計算する処理はなく、Stim の detector sampler 出力をそのまま使う。[根拠: `src/graph_representation.py`, `sample_syndromes`, lines 15-49]

- 最終ラウンドの扱い:
  - 明示的記述: `self.syndrome_mask` の時間軸長は `d_t + 1` であり、detector 座標の time 成分をそのまま配置する。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 163-179; `src/decoder.py`, `Decoder.stim_to_syndrome_3D`, lines 186-194]
  - 判定: 最終ラウンドを data-qubit readout から repo 自身が再構成するコードは未確認。もし最終検出イベントが data readout に由来するなら、その処理は Stim generator / detector sampler 側に委譲されている。

### 4.3 出力データ

- 明示的記述:
  - モデル出力は `num_classes` 次元の logits であり、現行設定では 1 ビットの `observable_flips` を予測する。[根拠: `src/gnn_models.py`, `GNN_7.__init__`, lines 15-42; `src/gnn_models.py`, `GNN_7.forward`, lines 44-61; `config_surface_codes_3_3.yaml`, 記載本文, lines 5-18]

- 非対応:
  - 物理エラー string の出力はない。[根拠: `src/decoder.py`, `Decoder.evaluate_test_set`, lines 231-245]
  - MWPM 用の事前計算 edge weight を外部へ返す API もない。`edge_attr` は内部の GNN 入力であり decode 結果ではない。[根拠: `src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 223-229]

### 4.4 運用形態

- 判定: logical readout-only decoding に該当する。
  - 明示的記述: 学習・評価の教師信号は `observable_flips` であり、補正後の物理 Pauli を構成しない。[根拠: `src/graph_representation.py`, `sample_syndromes`, lines 45-49; `src/decoder.py`, `Decoder.train`, lines 297-313]

- full correction / active correction:
  - 明示的記述: 物理補正演算を返すメソッドも、回路へ補正をフィードバックするメソッドもない。[根拠: `src/decoder.py`, `Decoder`, lines 20-418]
  - 判定: full correction ではない。active correction として回路へ即時フィードバック可能な形にもなっていない。

- Pauli frame update:
  - 推論: 予測された logical flip を外部で frame update に使うこと自体は概念上可能だが、repo はその更新器を実装していない。[根拠: `src/decoder.py`, `Decoder.evaluate_test_set`, lines 231-245]

## 5. Neural network 系アルゴリズムの対応

### 5.1 training / inference 対応

- training に対応している。
  - 明示的記述: `train_nn.py` が `Decoder.run()` を起動し、`run_training` が真なら `Decoder.train()` を呼ぶ。[根拠: `train_nn.py`, `main`, lines 11-22; `src/decoder.py`, `Decoder.run`, lines 405-418]

- inference / evaluation に対応している。
  - 明示的記述: `run_test` が真なら `Decoder.test()` を実行し、`resume_training` 時は保存済み重みをロードする。[根拠: `src/decoder.py`, `Decoder.load_trained_model`, lines 124-133; `src/decoder.py`, `Decoder.run`, lines 405-418]

### 5.2 合成訓練データの生成方法

- 明示的記述:
  - 訓練データは毎 batch オンライン生成であり、ファイルから読まない。[根拠: `src/decoder.py`, `Decoder.train`, lines 294-305]
  - データ生成は `sample_syndromes(batch_size, self.compiled_sampler, self.device)` で行われる。[根拠: `src/decoder.py`, `Decoder.train`, lines 294-299]
  - `sample_syndromes` は Stim sampler から `detection_events` / `observable_flips` をサンプルし、空 syndrome を除外する。[根拠: `src/graph_representation.py`, `sample_syndromes`, lines 12-24, 33-49]

- train error rate list の扱い:
  - 明示的記述: `compiled_sampler` が list の場合、各 error rate ごとに必要数までサンプルし、その後 interleave して混ぜる。[根拠: `src/graph_representation.py`, `sample_syndromes`, lines 6-32]
  - 推論: 訓練分布は単一 p ではなく、設定ファイルで指定した複数 p の mixture になっている。[根拠: `config_surface_codes_3_3.yaml`, 記載本文, lines 19-25; `src/graph_representation.py`, `sample_syndromes`, lines 6-32]

- 旧 artifact 群:
  - 判定: `perfect_stabilizers` / `repetition_code` の training code は未確認。モデル artifact はあるが、合成データ生成 script は現行 checkout には見当たらない。[根拠: `README.md`, 記載本文, lines 58-59; `train_nn.py`, `main`, lines 11-22; `src/decoder.py`, `Decoder.initialise_simulations`, lines 135-161]

## 6. ベンチマークの評価内容

### 6.1 現行ソースで実行可能な評価

- 評価対象は logical memory 実験であり、1-shot readout ではない。
  - 明示的記述: 使用する circuit generator は `surface_code:rotated_memory_z`、かつ `rounds=self.d_t` を持つ。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 137-145]
  - 推論: `memory_z` generator を repeated rounds で使うため、少なくとも repo の主タスクは repeated-syndrome を伴う logical memory decoding である。

- 評価指標は logical observable の classification accuracy である。
  - 明示的記述: `evaluate_test_set` は sigmoid しきい値 0.5 で二値化し、`prediction == target` を全 class 一致で数える。[根拠: `src/decoder.py`, `Decoder.evaluate_test_set`, lines 231-245]
  - 明示的記述: `Decoder.test` は batch ごとの `Accuracy` と trivial syndrome 数を表示する。[根拠: `src/decoder.py`, `Decoder.test`, lines 381-398]

- trivial syndrome の扱い:
  - 明示的記述: `sample_syndromes` は空 syndrome を除外しつつ、その数を `n_trivial_syndromes` として数える。[根拠: `src/graph_representation.py`, `sample_syndromes`, lines 18-24, 37-43]
  - 明示的記述: `evaluate_test_set` は最終 accuracy に `n_trivial_syndromes` を加算する。[根拠: `src/decoder.py`, `Decoder.evaluate_test_set`, lines 241-245]
  - 判定: 現行 benchmark の accuracy は「非自明 syndrome 上の分類精度」ではなく、「trivial shot を正答扱いで足し戻した全 shot 精度」である。

### 6.2 評価条件

- stabilizer round 数:
  - 明示的記述: round 数は設定ファイルの `d_t` に依存する。現行設定例は `d_t: 3`。[根拠: `config_surface_codes_3_3.yaml`, 記載本文, lines 19-25; `config_surface_codes_9_3.yaml`, 記載本文, lines 22-28]
  - 明示的記述: 保存済み circuit-level model ファイル名には `d_t_3`, `d_t_5`, `d_t_7`, `d_t_9`, `d_t_11` が含まれる。[根拠: `models/circuit_level_noise/`, ファイル一覧]

- logical Z 偏り:
  - 明示的記述: generator 名が `surface_code:rotated_memory_z` である。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 139-152]
  - 判定: 現行ソースで確認できる評価は logical Z memory タスクに偏っている。logical X memory の別実装は未確認。

- 誤りモデル前提:
  - 明示的記述: 現行実行系は circuit-level noise を使う。[根拠: `src/decoder.py`, `Decoder.initialise_simulations`, lines 137-158]
  - 判定: しきい値図や比較図を再生成する script は現行 checkout では未確認。README は figure 番号と対応 model directory を挙げるが、plot script / result table / raw benchmark log は含まれない。[根拠: `README.md`, 記載本文, lines 58-59]

### 6.3 artifact の存在と限界

- 明示的記述:
  - `models/circuit_level_noise/`, `models/perfect_stabilizers/`, `models/repetition_code/` には保存済み `.pt` がある。[根拠: 各ディレクトリのファイル一覧]

- 判定:
  - 「ベンチマーク結果が存在するか」という意味では、学習済みモデル artifact は存在する。
  - ただし、数値結果そのものを保存した表・図・ログは repo 内で未確認であり、何を何と比較した benchmark かを artifact だけから完全再構成することはできない。
