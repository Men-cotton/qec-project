# Repo Investigation: GNN Decoder Surface Code

最終更新: 2026-04-12

この文書は `NN-based/PLANS.md` の調査観点に合わせて、**「シンドロームグラフからマッチングを出力する過程」** に絞って整理する。各主張では `明示的記述` と `推論` を区別し、証拠がない項目は `未確認` と記す。

## 1. 対象となるグラフ構造と実装範囲

### 1.1 結論

- この repo は、**シンドロームグラフ上のマッチング処理を実装もラップもしていない**。
  - 明示的記述: README は本 repo を "Graph neural network decoder for the rotated surface code" と説明し、主要モジュールとして `decoder.py`, `gnn_models.py`, `graph_representation.py` を列挙している。matching solver や MWPM wrapper は列挙されていない。[根拠: `NN-based/GNN_decoder/README.md`, 記載本文 lines 1-2, 45-50]
  - 明示的記述: `Decoder.__init__` は `GNN_7` を生成し、`GNN_7.forward` は `GraphConv` と `global_mean_pool` と MLP を適用して出力 logits を返す。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.__init__`, lines 65-72; `NN-based/GNN_decoder/src/gnn_models.py`, `GNN_7.forward`, lines 44-61]
  - 明示的記述: 評価は `observable_flips` に対する二値分類精度であり、マッチされたノード対や選択エッジ集合は返さない。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.evaluate_test_set`, lines 231-245]
  - 明示的記述: 依存関係には `stim`, `torch`, `torch-geometric` はあるが、`pymatching`, `blossom`, `networkit` など matching backend は含まれない。[根拠: `NN-based/GNN_decoder/requirements.txt`, lines 1-53]

- したがって、この repo で確認できるのは **matching graph solver** ではなく、**detector event から GNN 入力グラフを構成する前処理** と **logical observable flip の推論** までである。
  - 明示的記述: `sample_syndromes` が `detection_events` と `observable_flips` をサンプルし、`get_batch_of_graphs` が kNN グラフを構築する。[根拠: `NN-based/GNN_decoder/src/graph_representation.py`, `sample_syndromes`, lines 4-49; `NN-based/GNN_decoder/src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 196-229]
  - 推論: `PLANS.md` が想定する「matching result の出力」は本 repo の API には存在しないため、以降の節では「存在する graph construction」と「存在しない matching stage」を分けて記述する。

### 1.2 サポートしている surface-code グラフの前提

- 明示的に使われるコードは `surface_code:rotated_memory_z` のみである。
  - 明示的記述: `Decoder.initialise_simulations` は `stim.Circuit.generated("surface_code:rotated_memory_z", ...)` を固定で呼ぶ。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.initialise_simulations`, lines 135-161]
  - 明示的記述: README も `rotated surface code` とだけ説明している。[根拠: `NN-based/GNN_decoder/README.md`, 記載本文 lines 1-2]

- XZZX, toric code, periodic boundary, hypergraph-based correlated decoder は確認できない。
  - 明示的記述: generator 名の分岐や code family の切替ロジックは `Decoder.initialise_simulations` に存在しない。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.initialise_simulations`, lines 135-161]
  - 推論: 現行設計は Stim の rotated planar memory circuit を固定利用する前提であり、toric/periodic 系や XZZX 専用 graph を対象にしていない。

- グラフは 3D 時空間 defect 配置を前提にする。
  - 明示的記述: detector 座標は `(x, y, t)` を持ち、`syndrome_mask` の時間軸長は `d_t + 1` である。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.initialise_simulations`, lines 163-179]
  - 明示的記述: `stim_to_syndrome_3D` は detector event を 3D syndrome grid に配置する。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.stim_to_syndrome_3D`, lines 181-194]

- boundary node は明示的には表現されない。
  - 明示的記述: `get_batch_of_graphs` がノード化するのは `np.nonzero(syndromes)` で得た defect のみであり、boundary node を追加する処理はない。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 196-223]
  - 推論: Stim circuit には境界が内在していても、repo 内の GNN 入力グラフでは boundary は独立ノードや終端として露出していない。

### 1.3 Capability Matrix

以下は **matching solver の能力表ではなく、この repo が実際に構築する syndrome/defect graph 前処理の能力表** である。

| Graph dimension (2D/3D) | Boundary node support | Weighted edges support | Hyperedge support (for correlated/Y errors) | Dynamic graph generation | Parallel matching support |
| --- | --- | --- | --- | --- | --- |
| 3D space-time defect graph。`(x, z, t)` を持つ | 非対応。boundary node を明示生成しない | 対応。`edge_attr = 1 / dist ** power` | 非対応。通常 edge のみ | 対応。各 batch ごとに `sample_syndromes` と `get_batch_of_graphs` を実行 | 非対応。matching stage 自体が存在しない |

## 2. グラフ構築とエッジ重みの計算

### 2.1 detector event の生成

- 明示的記述: `sample_syndromes` は compiled Stim detector sampler から `detection_events` と `observable_flips` を取得する。`separate_observables=True` が使われている。[根拠: `NN-based/GNN_decoder/src/graph_representation.py`, `sample_syndromes`, lines 15-17, 34-36]

- 明示的記述: 空 syndrome は除外され、除外数は `n_trivial_syndromes` として数えられる。[根拠: `NN-based/GNN_decoder/src/graph_representation.py`, `sample_syndromes`, lines 18-24, 37-43]

### 2.2 2D code-capacity と 3D circuit-level の違い

- 明示的記述: 現行コードで明示実装されているのは repeated rounds を持つ `surface_code:rotated_memory_z` の circuit 生成だけである。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.initialise_simulations`, lines 137-158]

- 明示的記述: ノイズは `after_clifford_depolarization`, `after_reset_flip_probability`, `before_measure_flip_probability`, `before_round_data_depolarization` に同じ確率で注入される。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.initialise_simulations`, lines 142-145, 155-158]

- 推論: code-capacity 専用の 2D matching graph や phenomenological 専用分岐は少なくとも repo 内には存在しない。実装されているのは measurement error を含む 3D detector-event 生成経路のみである。

### 2.3 syndrome grid と defect graph への変換

- 明示的記述: `stim_to_syndrome_3D` は nonzero detector に対し、X stabilizer を `1`、Z stabilizer を `3` に変換する。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.stim_to_syndrome_3D`, lines 181-194]

- 明示的記述: `get_batch_of_graphs` は nonzero 位置だけを defect node にし、X defect なら feature 0、Z defect なら feature 1 を立て、残りに座標を入れる。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 199-218]

- 推論: 実際のモデル入力 feature は `[is_X, is_Z, x, z, t]` の 5 次元である。`node_features` 6 列のうち batch index 列を除いた `x_cols = [0, 1, 3, 4, 5]` が `x` へ渡されるため。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 207-220]

### 2.4 エッジ重み

- 明示的記述: 辺は `knn_graph(pos, self.m_nearest_nodes, batch=batch_labels)` で張られる。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 220-223]

- 明示的記述: edge weight はノード間ユークリッド距離 `dist` に対して `edge_attr = 1 / dist ** self.power` で計算される。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 225-227]

- 推論: これは matching cost に典型的な `-log p` や整数化重みではなく、GNN message passing 用の幾何学的 edge attribute である。

### 2.5 相関誤り / Y 誤り

- 明示的記述: repo 側が保持するのは detector event と stabilizer type のみであり、Y 誤り専用ラベルや hyperedge 構造は存在しない。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.stim_to_syndrome_3D`, lines 181-194; `NN-based/GNN_decoder/src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 196-229]

- 推論: 相関誤りが元の回路に含まれていても、本 repo では Stim が吐いた detector event に畳み込まれ、graph 構造としては通常の pairwise kNN edge にしか現れない。

## 3. マッチングアルゴリズムの概要

### 3.1 結論

- 明示的記述: コアアルゴリズムは MWPM, Union-Find, Blossom ではなく、`GraphConv` ベースの GNN と MLP による graph classification である。[根拠: `NN-based/GNN_decoder/src/gnn_models.py`, `GNN_7`, lines 6-61]

- 明示的記述: `Decoder` は `GNN_7` を用いて `observable_flips` を学習・評価する。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.__init__`, lines 65-72; `NN-based/GNN_decoder/src/decoder.py`, `Decoder.train`, lines 297-314; `NN-based/GNN_decoder/src/decoder.py`, `Decoder.evaluate_test_set`, lines 231-245]

- 明示的記述: matching library 依存は `requirements.txt` に見当たらない。[根拠: `NN-based/GNN_decoder/requirements.txt`, lines 1-53]

### 3.2 アルゴリズムとして確認できる処理

1. `stim.Circuit.generated(...)` で rotated surface-code memory circuit を作る。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.initialise_simulations`, lines 137-158]
2. `sample_syndromes` で `detection_events` と `observable_flips` をサンプルする。[根拠: `NN-based/GNN_decoder/src/graph_representation.py`, `sample_syndromes`, lines 4-49]
3. `stim_to_syndrome_3D` で 3D syndrome grid へ変換する。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.stim_to_syndrome_3D`, lines 181-194]
4. `get_batch_of_graphs` で defect node と kNN edge を作る。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 196-229]
5. `GNN_7.forward` で graph embedding を作り、logical observable flip の logits を返す。[根拠: `NN-based/GNN_decoder/src/gnn_models.py`, `GNN_7.forward`, lines 44-61]

- 推論: `PLANS.md` の意味での「matching を解くコアアルゴリズム」は未実装であり、ここでの graph processing は GNN 推論の前処理で止まる。

## 4. 入出力インターフェースとデータ構造

### 4.1 入力データ

- グラフ入力:
  - 明示的記述: `get_batch_of_graphs(self, syndromes)` は `x`, `edge_index`, `batch_labels`, `edge_attr` を返す。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 196-229]
  - 明示的記述: `x` と `edge_attr` は Torch tensor、`edge_index` は `knn_graph` が返す PyG 形式、`batch_labels` は graph ごとの batch index tensor である。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 218-229]
  - 推論: repo は NetworkX や CSR を使わず、PyTorch Geometric 向け tensor 表現を直接構築している。

- syndrome 入力:
  - 明示的記述: `sample_syndromes(n_shots, compiled_sampler, device)` は `detection_events`, `observable_flips`, `n_trivial_syndromes` を返す。[根拠: `NN-based/GNN_decoder/src/graph_representation.py`, `sample_syndromes`, lines 4-49]
  - 明示的記述: `detection_events` は NumPy array、`observable_flips` は `torch.tensor(..., dtype=torch.float32)` に変換される。[根拠: `NN-based/GNN_decoder/src/graph_representation.py`, `sample_syndromes`, lines 45-49]

### 4.2 出力データ

- 明示的記述: `GNN_7.forward(self, x, edge_index, batch, edge_attr)` の返り値は `output` logits である。[根拠: `NN-based/GNN_decoder/src/gnn_models.py`, `GNN_7.forward`, lines 44-61]

- 明示的記述: `evaluate_test_set` は sigmoid 後に 0.5 しきい値で二値化し、`observable_flips` と比較する。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.evaluate_test_set`, lines 231-245]

- 明示的記述: `num_classes: 1` なので、現行設定では 1 個の logical observable flip を予測する。[根拠: `NN-based/GNN_decoder/config_surface_codes_3_3.yaml`, lines 5-25; `NN-based/GNN_decoder/config_surface_codes_9_3.yaml`, lines 5-28]

- 結論:
  - マッチング結果のノード対リスト: 非対応
  - 選択エッジ集合: 非対応
  - 物理 correction string: 非対応
  - logical observable flip の二値予測: 対応

### 4.3 コア関数・クラスのシグネチャ

- `sample_syndromes(n_shots, compiled_sampler, device) -> detection_events, observable_flips, n_trivial_syndromes`
  - 根拠: `NN-based/GNN_decoder/src/graph_representation.py`, `sample_syndromes`, lines 4-49

- `Decoder.stim_to_syndrome_3D(self, detection_events_list) -> syndrome_3D`
  - 根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.stim_to_syndrome_3D`, lines 181-194

- `Decoder.get_batch_of_graphs(self, syndromes) -> x, edge_index, batch_labels, edge_attr`
  - 根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 196-229

- `GNN_7.forward(self, x, edge_index, batch, edge_attr) -> output`
  - 根拠: `NN-based/GNN_decoder/src/gnn_models.py`, `GNN_7.forward`, lines 44-61

## 5. Neural network 系アルゴリズムの適用

- 明示的記述: neural network は使われているが、用途は **matching edge の予測** ではなく **graph 全体から logical observable flip を直接分類すること** である。[根拠: `NN-based/GNN_decoder/src/gnn_models.py`, `GNN_7`, lines 6-61; `NN-based/GNN_decoder/src/decoder.py`, `Decoder.evaluate_test_set`, lines 231-245]

- 明示的記述: graph / syndrome は次の流れで tensor 化される。
  - `detection_events` を 3D syndrome grid へ配置する。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.stim_to_syndrome_3D`, lines 181-194]
  - nonzero syndrome を defect node に変換する。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 199-212]
  - node feature `[is_X, is_Z, x, z, t]` を `x` に、kNN edge を `edge_index` に、距離重みを `edge_attr` に入れる。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.get_batch_of_graphs`, lines 207-229]

- 推論: `PLANS.md` の「GNN による matching edge の直接予測」には該当しない。現行モデルは edge classification でも pairing prediction でもなく、graph-level binary classification である。

## 6. マッチング処理のパフォーマンス・ベンチマーク

- 明示的記述: 現行の評価指標は matching solver の実行時間や matching cost ではなく、`observable_flips` の classification accuracy である。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.evaluate_test_set`, lines 231-245; `NN-based/GNN_decoder/src/decoder.py`, `Decoder.test`, lines 370-402]

- 明示的記述: `train()` は `Sampling and Graphing`, `Fitting`, `Writing` の時間を出力する。[根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.train`, lines 270-273, 362-368]

- 推論: ここで測られている時間は GNN 学習パイプライン全体の粗い内訳であり、matching solver 単体の runtime benchmark ではない。

- 明示的記述: 設定ファイルは `dataset_size`, `validation_set_size`, `test_set_size`, `acc_test_size` などを与えるが、matching graph size に対する scaling 指標やメモリ計測は定義しない。[根拠: `NN-based/GNN_decoder/config_surface_codes_3_3.yaml`, lines 26-41; `NN-based/GNN_decoder/config_surface_codes_9_3.yaml`, lines 29-44]

- 結論:
  - matching solver としてのベンチマーク: 非対応
  - GNN decoder の分類精度評価: 対応
  - matching cost / 厳密解とのコスト差 / メモリ使用量: 未確認ではなく、少なくとも現行コードには実装なし

## 最終結論

- この subrepo は、surface-code の detector event から **3D defect kNN graph** を動的生成し、それを **GNN に入力して logical observable flip を直接予測する** 実装である。
  - 根拠: `NN-based/GNN_decoder/src/decoder.py`, `Decoder.initialise_simulations`, `Decoder.stim_to_syndrome_3D`, `Decoder.get_batch_of_graphs`, lines 135-229; `NN-based/GNN_decoder/src/gnn_models.py`, `GNN_7.forward`, lines 44-61

- `PLANS.md` が対象とする意味での **「シンドロームグラフからマッチングを出力する過程」** は、この repo には存在しない。
  - 根拠: `NN-based/GNN_decoder/README.md`, lines 1-2, 45-50; `NN-based/GNN_decoder/src/decoder.py`, `Decoder.evaluate_test_set`, lines 231-245; `NN-based/GNN_decoder/requirements.txt`, lines 1-53
