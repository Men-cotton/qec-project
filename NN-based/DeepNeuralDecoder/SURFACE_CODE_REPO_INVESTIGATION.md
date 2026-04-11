# DeepNeuralDecoder Surface-Code Matching Investigation

この文書は `NN-based/PLANS.md` の指示に従い、DeepNeuralDecoder repo を「シンドロームグラフからマッチングを出力する過程」に限定して調査した結果を整理したものである。

## 0. 結論

- `結論`
  - この repo は、surface code 向けの `syndrome graph / detection graph` を構築し、その上で `matching` を解く実装またはラッパを持っていない。
  - したがって、`MWPM` などの「グラフ上のマッチング処理」を調査対象とする限り、本 repo の surface-code 実装は `未対応` と判断するのが妥当である。

- `明示的記述`
  - surface code の Python 側エントリポイントは `Trainer/Run.py` の `run_pickler` / `run_benchmark` であり、`Surface1EC` 系として生成されるのは `LookUpSurface1EC` と `PureErrorSurface1EC` のみである。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/Run.py` / `run_pickler`, `run_benchmark`
    - `NN-based/DeepNeuralDecoder/Trainer/ModelSurface1EC.py` / `LookUpSurface1EC`, `PureErrorSurface1EC`
  - base decoder のコア処理は、`syndrome -> correctionMat[key][index]` の table lookup か、`syndrome -> syn * T[key]` の線形写像である。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/Model.py` / `lookup_correction`, `pure_correction`
  - surface code 用 `Spec` は `L`, `G`, `T`, `correctionMat` という二値行列を保持するだけで、隣接リスト、edge list、boundary node 集合、edge weight 配列のような graph object を持たない。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
    - `NN-based/DeepNeuralDecoder/Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`
  - README はこの repo を「CSS codes 向け deep learning framework」と説明し、surface code についても `Pure-Error` と `LookUp` を base decoder として列挙している。matching solver への言及はない。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/README.md` / 該当関数・クラスなし

- `推論`
  - この repo の surface decoder は「matching graph を解く decoder」ではなく、「representative syndrome を選んで固定補正を当て、その後に NN で logical frame を推定する decoder」である。
  - 以降の各節では、PLANS の項目に沿って `未対応` の根拠と、代わりに実装されている処理を整理する。

## 1. 対象となるグラフ構造と実装範囲

### 1.1 マッチング処理の有無

- `明示的記述`
  - `Surface1EC` は入力データを `synX`, `synZ`, `errX`, `errZ` に分解し、`choose_syndrome` で複数ラウンドの syndrome から 1 つの代表 syndrome を選ぶ。ここでは graph 構築は行われない。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/ModelSurface1EC.py` / `get_data`, `choose_syndrome`
  - `syn_from_generators` は `err * G[perp(key)]^T mod 2` により syndrome を生成するだけで、ノードやエッジの生成処理ではない。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/Model.py` / `syn_from_generators`
  - `lookup_correction` は syndrome bit 列を整数 index に変換して `correctionMat` を参照する。`pure_correction` は syndrome に `T[key]` を掛ける。いずれも graph search や matching solver 呼び出しを含まない。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/Model.py` / `lookup_correction`, `pure_correction`
    - `NN-based/DeepNeuralDecoder/Trainer/util.py` / `vec_to_index`

- `証拠なし`
  - `MWPM`
  - `Blossom`
  - `PyMatching`
  - `NetworkX` ベースの syndrome graph
  - `Union-Find`
  - hypergraph matching

- `判定`
  - この repo は surface code に対して「シンドロームグラフ上のマッチング処理」を `実装していない`。

### 1.2 どの surface code グラフを前提としているか

- `明示的記述`
  - README は `Rotated surface code single EC` と記述する。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/README.md` / 該当関数・クラスなし
  - surface code generator は Matlab 側で `SurfaceCodeCircuitGenerator(d)` を使い、`d=3` と `d=5` のデータセットを生成する。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `SurfaceCodeCircuitGenerator`
    - `NN-based/DeepNeuralDecoder/Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `SurfaceCodeCircuitGenerator`
  - Python 側 spec は `SurfaceD3` と `SurfaceD5` の固定実装のみを持つ。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/Run.py` / main ブロック
    - `NN-based/DeepNeuralDecoder/Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
    - `NN-based/DeepNeuralDecoder/Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`

- `推論`
  - 前提コード族は rotated planar CSS surface code だが、その幾何は `matching graph` としては Python 側に持ち込まれていない。
  - Python 側で使われるのは stabilizer generator 行列 `G`、logical operator `L`、補正行列 `T`、lookup table `correctionMat` であり、「ノード集合とエッジ集合」ではない。

### 1.3 グラフ・マッチング capability matrix

| Surface-code path in repo | Graph dimension (2D/3D) | Boundary node support | Weighted edges support | Hyperedge support (for correlated/Y errors) | Dynamic graph generation | Parallel matching support |
| --- | --- | --- | --- | --- | --- | --- |
| `LookUpSurface1EC` | `未対応` | `未対応` | `未対応` | `未対応` | `未対応` | `未対応` |
| `PureErrorSurface1EC` | `未対応` | `未対応` | `未対応` | `未対応` | `未対応` | `未対応` |
| NN overlay (`FF` / `RNN` / `3DCNN` / `Ch3DCNN`) | `未対応` | `未対応` | `未対応` | `未対応` | `未対応` | `未対応` |

- `補足`
  - ここでの `未対応` は、「surface-code support 自体が無い」という意味ではなく、「matching graph をデータ構造として保持・生成・最適化する機能が無い」という意味である。

## 2. グラフ構築とエッジ重みの計算

### 2.1 シンドロームグラフ構築の有無

- `明示的記述`
  - Python decoder 側には graph builder が存在せず、compact データから読み込まれるのは各ラウンドの syndrome bit 列と error bit 列である。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/ModelSurface1EC.py` / `get_data`
    - `NN-based/DeepNeuralDecoder/Data/Compact/Surface_1EC_D3/compressor.py` / `run`
    - `NN-based/DeepNeuralDecoder/Data/Compact/Surface_1EC_D5/compressor.py` / `run`
  - d=3 では `choose_syndrome` が 3 ラウンドの syndrome から代表ラウンドを選ぶ。d=5 でも同様に代表ラウンド index を決める。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/ModelSurface1EC.py` / `choose_syndrome`

- `推論`
  - repo は repeated syndrome history を `time-like edge を持つ 3D decoding graph` に変換していない。
  - measurement error は「時間方向グラフの matching」で処理されず、「代表 syndrome の選択規則」で吸収される。

### 2.2 ノイズモデルの反映方法

- `明示的記述`
  - Matlab generator の `ErrorGenerator` と `depolarizingSimulator` は、storage、measurement、prep、CNOT fault を含む circuit-level stochastic Pauli noise を注入し、その結果として syndrome と error string を出力する。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `ErrorGenerator`, `depolarizingSimulator`
    - `NN-based/DeepNeuralDecoder/Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `ErrorGenerator`, `depolarizingSimulator`
  - `PropagationStatePrepArb` は error type `1=X`, `2=Z`, `3=Y` を扱い、Y では X/Z の両成分を立てる。CNOT fault も control/target に展開される。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `PropagationStatePrepArb`
    - `NN-based/DeepNeuralDecoder/Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `PropagationStatePrepArb`

- `明示的記述`
  - しかし Python decoder 側には `p` や bias ratio から edge weight を計算する処理がない。補正処理は `lookup_correction` と `pure_correction` に固定されている。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/Model.py` / `lookup_correction`, `pure_correction`

- `判定`
  - `edge weight の計算・保持`: `未対応`
  - `log-likelihood / integer weight への丸め`: `未対応`
  - `code-capacity 用 2D graph と circuit-level 用 3D graph の切替`: `未対応`

### 2.3 Y 誤り・相関誤りの扱い

- `明示的記述`
  - generator は Y fault と 2-qubit CNOT fault を生成する。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Data/Generator/Surface_1EC_D3/SurfaceCodeTrainingSetd3.m` / `PropagationStatePrepArb`, `ErrorGenerator`
    - `NN-based/DeepNeuralDecoder/Data/Generator/Surface_1EC_D5/SurfaceCodeTrainingSetd5.m` / `PropagationStatePrepArb`, `ErrorGenerator`
  - decoder の内部表現は常に `err_keys=['X','Z']` の 2 チャネルである。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/_SurfaceD3Lookup.py` / `Spec.__init__`
    - `NN-based/DeepNeuralDecoder/Trainer/_SurfaceD5Lookup.py` / `Spec.__init__`

- `推論`
  - Y 誤りや相関誤りは matching graph 上の hyperedge としては扱われず、X/Z 成分へ射影された syndrome / error string に畳み込まれる。

## 3. マッチングアルゴリズムの概要

### 3.1 実装されているコアアルゴリズム

| Path | コア処理 | matching か |
| --- | --- | --- |
| `LookUpSurface1EC` | syndrome index を `correctionMat` に引く table lookup | `否` |
| `PureErrorSurface1EC` | syndrome に `T[key]` を掛ける固定線形写像 | `否` |
| NN overlay | syndrome history から logical bit を 2-class 分類 | `否` |

- `明示的記述`
  - `LookUpSurface1EC.init_rec` は `abstract_init_rec(..., self.lookup_correction)` を呼ぶ。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/ModelSurface1EC.py` / `LookUpSurface1EC.init_rec`
  - `PureErrorSurface1EC.init_rec` は `abstract_init_rec(..., self.pure_correction)` を呼ぶ。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/ModelSurface1EC.py` / `PureErrorSurface1EC.init_rec`
  - `num_logical_fault` は NN 出力 bit を `pred[key][i] * self.spec.L[key]` として logical operator 適用有無に使う。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/Model.py` / `num_logical_fault`

### 3.2 外部ライブラリ依存

- `明示的記述`
  - NN 実装は TensorFlow 1.x に依存する。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/README.md` / 該当関数・クラスなし
    - `NN-based/DeepNeuralDecoder/Trainer/Networks.py` / 複数関数

- `証拠なし`
  - PyMatching
  - Blossom V
  - Lemon
  - Kolmogorov Blossom
  - custom blossom implementation

- `判定`
  - matching solver 依存は確認できない。

## 4. 入出力インターフェースとデータ構造

### 4.1 入力データ（グラフ）

- `結論`
  - `シンドロームグラフ自体を入力する API は存在しない。`

- `明示的記述`
  - `Surface1EC.get_data(path)` はテキスト行を `synX`, `synZ`, `errX`, `errZ` の `np.matrix` に変換する。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/ModelSurface1EC.py` / `get_data`
  - d=3 compact data は `synX1synX2synX3 errX1errX2errX3 synZ1synZ2synZ3 errZ1errZ2errZ3` 形式で保存される。d=5 も同型で syndrome 部と error 部を連結する。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Data/Compact/Surface_1EC_D3/compressor.py` / `run`
    - `NN-based/DeepNeuralDecoder/Data/Compact/Surface_1EC_D5/compressor.py` / `run`

### 4.2 入力データ（シンドローム）

- `明示的記述`
  - `self.syn['X']`, `self.syn['Z']` は flat な binary matrix であり、graph node ID の list ではない。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/ModelSurface1EC.py` / `init_syn`
  - FF では placeholder は `[None, spec.input_size]`。RNN ではその flat vector を `[-1, spec.num_epochs, spec.lstm_input_size]` に reshape する。3DCNN では `[-1, spec.num_syn, spec.syn_w[key], spec.syn_h[key], 1]` またはチャネル結合版に reshape する。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/Model.py` / `train`
    - `NN-based/DeepNeuralDecoder/Trainer/Networks.py` / `ff_cost`, `rnn_cost`, `surface_conv3d_cost`, `surface_channeled_conv3d_cost`

- `推論`
  - matching solver に defect node list を渡すのではなく、NN には syndrome history 全体をテンソルとして渡している。

### 4.3 出力データ

- `明示的記述`
  - base decoder の内部出力は `self.rec[key] = raw_data['err'+key] + abs_corr(rep_syn[key], key)` である。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/ModelSurface1EC.py` / `abstract_init_rec`
  - NN の出力は `predict[key] = tf.argmax(logits[key], 1)` の 0/1 class である。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/Networks.py` / `ff_cost`, `rnn_cost`, `surface_conv3d_cost`, `surface_channeled_conv3d_cost`
  - 最終評価ではこの 0/1 を logical operator 適用有無として解釈する。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/Model.py` / `num_logical_fault`

- `判定`
  - `matching pair list`: `未対応`
  - `selected edge list`: `未対応`
  - `physical correction bitstring`: `base decoder の内部状態としてのみ存在`
  - `logical frame bit`: `対応`

### 4.4 コア関数・クラスのシグネチャ

- `Surface1EC.get_data(path)`
  - 役割: compact text を `synX/synZ/errX/errZ` の `np.matrix` に変換する。
  - 根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/ModelSurface1EC.py` / `get_data`

- `Surface1EC.choose_syndrome(self, syn)`
  - 役割: 複数ラウンド syndrome から代表 syndrome と index を返す。
  - 戻り値: `(syn_dic[syndrome_index], syndrome_index)`
  - 根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/ModelSurface1EC.py` / `choose_syndrome`

- `Model.lookup_correction(self, syn, key)`
  - 役割: syndrome を table lookup で correction vector に写像する。
  - 根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/Model.py` / `lookup_correction`

- `Model.pure_correction(self, syn, key)`
  - 役割: syndrome を線形写像で correction vector に写像する。
  - 根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/Model.py` / `pure_correction`

- `Model.train(self, param, tune=False, save=False, save_path=None)`
  - 役割: NN を学習し、prediction と test offset を返す。
  - 戻り値: `prediction, t_beg` または tuning cost
  - 根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/Model.py` / `train`

## 5. Neural network 系アルゴリズムの適用

### 5.1 NN が使われている場所

- `明示的記述`
  - `FF`, `RNN`, `3DCNN`, `Ch3DCNN` が surface code 用ネットワークとして実装されている。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/Networks.py` / `ff_cost`, `rnn_cost`, `surface_conv3d_cost`, `surface_channeled_conv3d_cost`
  - `init_log_1hot` は residual error から 2-class logical label を作る。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/Model.py` / `init_log_1hot`

- `推論`
  - NN は edge weight 推定器でも matching edge 予測器でもない。
  - 実際には `base decoder 後に logical operator を反転させるべきか` を分類している。

### 5.2 テンソル変換

| NN path | 入力テンソル化 | 出力 |
| --- | --- | --- |
| `FF` | syndrome history の flat vector | 各 `key` について 2-class logical bit |
| `RNN` | `[-1, num_epochs, lstm_input_size]` | 各 `key` について 2-class logical bit |
| `3DCNN` | `[-1, num_syn, syn_w[key], syn_h[key], 1]` | 各 `key` について 2-class logical bit |
| `Ch3DCNN` | X/Z を別 reshape 後に channel concat | X/Z 各 head の 2-class logical bit |

- `判定`
  - `graph tensor` への変換: `未対応`
  - `edge score` 出力: `未対応`
  - `matching pair` 直接予測: `未対応`

## 6. マッチング処理のパフォーマンス・ベンチマーク

### 6.1 何を評価しているか

- `明示的記述`
  - `run_benchmark` は学習後に `m.num_logical_fault(prediction, test_beg)` を評価する。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Trainer/Run.py` / `run_benchmark`
    - `NN-based/DeepNeuralDecoder/Trainer/Model.py` / `num_logical_fault`
  - plot script の軸は `Physical fault rate` と `Logical fault rate` である。根拠ファイル/関数:
    - `NN-based/DeepNeuralDecoder/Reports/poly_plot.py` / `plot_results`
    - `NN-based/DeepNeuralDecoder/Reports/simple_plot.py` / `plot_results`

- `判定`
  - 既存 benchmark は「matching solver としての性能」ではなく、「logical fault rate 改善」を評価している。

### 6.2 matching solver 指標の有無

- `証拠なし`
  - matching 実行時間
  - matching 用メモリ使用量
  - graph size に対する scaling
  - defect density に対する scaling
  - exact matching cost との差分

- `結論`
  - matching solver 性能を測るベンチマークは、この repo では確認できない。

## 7. 最終整理

- `最終判定`
  - この repo は surface code の repeated syndrome データを扱うが、それを syndrome graph に変換して matching を解く実装ではない。
  - 実装されているのは:
    - `representative syndrome` の選択
    - `lookup table` または `pure-error` 線形写像による base correction
    - residual に対する `logical frame` の NN 予測

- `未対応理由の整理`
  - `MWPM / graph decoder が単に未実装か`
    - 判定: `未実装`
    - 理由: Python 側 decoder API が最初から table/linear-map/NN classification を前提に設計されている。
  - `設計上も対象外か`
    - 判定: `かなり強い`
    - 理由: README が CSS-code deep learning framework を掲げ、surface code でも base decoder を `Pure-Error` / `LookUp` に限定しているため、repo の主眼は matching ではなく neural logical decoding にある。

- `実務上の読み替え`
  - 「シンドロームグラフからマッチングを出力する過程」を調べたい場合、この repo は主要対象としては不適切である。
  - 調査価値があるのはむしろ、「repeated syndrome history から representative syndrome を選び、base decoder + NN で logical fault rate を下げる過程」である。
