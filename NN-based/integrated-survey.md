# NN-based 各 repo の survey 結果統合メモ

## 対象にした survey markdown

- `neural_network_decoder/repo_investigation.md`
- `astra/repo_surface_code_investigation.md`
- `graphqec-paper/repo_surface_code_investigation.md`
- `QEC_GNN-RNN/repo_surface_code_investigation.md`
- `GNN_decoder/repo_investigation_surface_code.md`
- `VNA-Decoder/surface_code_repo_investigation.md`
- `DeepNeuralDecoder/SURFACE_CODE_REPO_INVESTIGATION.md`

注:
- `DeepNeuralDecoder` だけ survey ファイル名が他と異なるが、内容は同じく repo 調査結果 markdown なので対象に含めた。
- 以下は各 survey markdown の記述を比較しやすいように再整理したものであり、ここで新たに raw code を再解釈したものではない。

## 全体像

- 7 repo すべてが surface code を対象にしている。
- ただし「同じ surface code」とはいえ、実際に扱う対象はかなり異なる。
- 最も多いのは `single logical qubit` の memory/readout 系であり、`multi-patch`, `lattice surgery`, `active correction` を明示的に扱う repo は確認されなかった。
- code family は大半が `rotated planar CSS surface code` に寄っている。
- 例外は `graphqec-paper` で、generic rotated patch ではなく `Sycamore` 実験データに強く結び付いた surface-code 実装である。
- `neural_network_decoder` は surface code 対応自体は明示されるが、patch variant や境界条件の明示度が最も低い。

## まず押さえるべき共通点

- 多くの repo が `単一 logical qubit` を前提にしている。
- `XZZX` や lattice surgery を主対象にした実装は、survey markdown の範囲では確認されなかった。
- benchmark の主眼は「logical memory / logical observable 判定」であり、実機フィードバック込みの online active correction ではない。
- neural decoder を含む repo が多いが、出力は一様ではない。
- repeated rounds と measurement error を本格的に扱う repo と、single-shot code-capacity 前提の repo が明確に分かれる。

## repo 横断比較

| repo | surface-code の実装スコープ | repeated rounds / measurement error | benchmark の主ノイズ前提 | 主デコーダと出力 | 位置づけ |
| --- | --- | --- | --- | --- | --- |
| `neural_network_decoder` | surface code 対応は明示。variant は未確認 | repeated rounds はあり。measurement error は未確認 | repo 内では固定されず、外部 DB 依存 | LSTM 系。出力は `parity` 1 ビット | 外部生成データを受ける parity predictor |
| `astra` | `RotatedPlanar2DCode` ベースの rotated planar CSS | repeated rounds なし。measurement error なし | code-capacity、single-shot、`X/Z/XZ/DP` 切替 | GNN 単体または GNN+MWPM/BP-OSD。物理 recovery を推定 | single-shot full-correction 寄り |
| `graphqec-paper` | `SycamoreSurfaceCode`。実験データ依存の CSS surface code | repeated rounds あり。measurement error あり | detector-level / DEM ベース。実験データ評価もあり | BPOSD, PyMatching, SlidingWindowBPOSD, GraphRNN。出力は logical observable | 実験データ・DEM 中心の統合基盤 |
| `QEC_GNN-RNN` | `surface_code:rotated_memory_z/x` | repeated rounds あり。measurement error あり | circuit-level | MWPM baseline と GNN+GRU。出力は logical observable 1 ビット | circuit-level memory task の logical predictor |
| `GNN_decoder` | `surface_code:rotated_memory_z` | repeated rounds あり。measurement error あり | circuit-level | GNN。出力は `observable_flips` | circuit-level memory task の graph classifier |
| `VNA-Decoder` | square rotated planar CSS patch | repeated rounds なし。measurement error なし | code-capacity、single-shot | EWD と VNA-RNN。等価類分布や class 選択 | single-shot generative / variational decoder |
| `DeepNeuralDecoder` | rotated planar CSS、実質 `d=3,5` 固定 | repeated rounds あり。measurement error あり | benchmark データ生成は circuit-level depolarizing | lookup / pure correction と NN。base は residual physical error、NN stage は X/Z logical bit | 古典補正 + NN logical stage の二段構成 |

## 共通点と相違点が最も出る軸

### 1. どの surface code を見ているか

- `rotated planar CSS` が主流:
  - `astra`
  - `QEC_GNN-RNN`
  - `GNN_decoder`
  - `VNA-Decoder`
  - `DeepNeuralDecoder`
- `graphqec-paper` は同じ surface-code 系でも `Sycamore` 実験データに依存する特殊化された実装で、generic rotated patch の survey とは性格が違う。
- `neural_network_decoder` は surface code 向けであることは明示されるが、`rotated/standard/toric/XZZX` などの切り分けは survey 上では確定していない。

### 2. single-shot か repeated-syndrome か

- `single-shot / code-capacity` 側:
  - `astra`
  - `VNA-Decoder`
- `repeated rounds + measurement error` 側:
  - `graphqec-paper`
  - `QEC_GNN-RNN`
  - `GNN_decoder`
  - `DeepNeuralDecoder`
- `中間的で不透明`:
  - `neural_network_decoder`
  - 時系列 `events` は扱うが、benchmark の具体的ノイズモデルは外部 DB に隠れている。

### 3. 出力が何か

- `logical/parity 直接予測`:
  - `neural_network_decoder`
  - `graphqec-paper`
  - `QEC_GNN-RNN`
  - `GNN_decoder`
- `physical recovery を推定して residual error を評価`:
  - `astra`
- `logical equivalence class の分布比較`:
  - `VNA-Decoder`
- `base correction と logical NN stage の併用`:
  - `DeepNeuralDecoder`

この軸が重要なのは、同じ「decoder」と書かれていても、repo ごとに最終出力の意味がかなり違うためである。特に `astra` は full-correction 寄りで、`QEC_GNN-RNN` や `GNN_decoder` は logical observable 判定寄りである。

### 4. 古典 decoder と neural decoder の関係

- `neural-only に近い`:
  - `neural_network_decoder`
  - `GNN_decoder`
- `neural + classical baseline/2段目補正`:
  - `astra`
  - `QEC_GNN-RNN`
  - `graphqec-paper`
  - `DeepNeuralDecoder`
- `非標準な generative / variational 系`:
  - `VNA-Decoder`

## repo のまとまり

### A. single-shot rotated planar を解く repo

- `astra`
- `VNA-Decoder`

共通点:
- code-capacity 前提
- repeated rounds を持たない
- measurement error を扱わない

相違点:
- `astra` は GNN と MWPM/OSD の接続が主で、物理 recovery を出す。
- `VNA-Decoder` は EWD や VNA-RNN で等価類分布を比較する、より統計力学寄りの構成である。

### B. circuit-level memory task を解く repo

- `QEC_GNN-RNN`
- `GNN_decoder`
- `DeepNeuralDecoder`

共通点:
- repeated syndrome rounds を持つ
- measurement error を含む
- benchmark は logical memory / logical observable 判定寄り

相違点:
- `QEC_GNN-RNN` は GNN+GRU と MWPM baseline を並置する。
- `GNN_decoder` は graph classifier として比較的単純で、`observable_flips` の二値分類が中心である。
- `DeepNeuralDecoder` は lookup / pure correction を前段に置き、その上に FF/RNN/3DCNN などを重ねる二段構成で、NN 単独 decoder とは少し性格が違う。

### C. 実験データ・DEM を中心にした repo

- `graphqec-paper`

特徴:
- `SycamoreSurfaceCode` に特化
- synthetic code-capacity benchmark より、`detector error model` と `実験 detection events` を扱う点が中心
- classical decoder と neural decoder の両方を同じ surface-code backend に載せて比較できる

### D. 外部データ依存の parity predictor

- `neural_network_decoder`

特徴:
- 時系列 `events` と `final syndrome increment` を受けて parity を直接予測する
- repo 内で code variant やノイズモデルが最も明示されていない
- 他 repo よりも「surface-code simulator 内蔵 repo」というより「surface-code 由来 dataset を受ける sequence model」に近い

## 重要な差分を短く言い切ると

- `astra` と `VNA-Decoder` は single-shot 系、`QEC_GNN-RNN` と `GNN_decoder` と `DeepNeuralDecoder` は repeated-round 系、`graphqec-paper` は実験データ / DEM 系である。
- `astra` は物理 recovery 推定まで踏み込むが、`QEC_GNN-RNN` と `GNN_decoder` は logical observable 判定に寄っている。
- `DeepNeuralDecoder` は NN だけで完結せず、lookup / pure correction を前段に持つ点が独特である。
- `graphqec-paper` は generic rotated planar benchmark 集ではなく、`Sycamore` surface-code path を中核に持つ点で他と大きく異なる。
- `neural_network_decoder` は surface code decoder ではあるが、比較軸の多くが外部 DB 依存で、repo 単体で確定できる範囲が最も狭い。

## この survey から見える実務上の使い分け

- `single-shot rotated planar` を比較したいなら、まず `astra` と `VNA-Decoder` を見るのが自然。
- `circuit-level repeated-syndrome memory` を比較したいなら、`QEC_GNN-RNN`, `GNN_decoder`, `DeepNeuralDecoder` が主対象になる。
- `実験データ` や `DEM` ベース評価まで含めて見たいなら、`graphqec-paper` が最も異質かつ情報量が多い。
- `logical parity 直接予測` の最小構成を見たいなら、`neural_network_decoder` は比較対象として有用だが、surface-code 幾何やノイズ前提の明示性は低い。

## 結論

- 全 repo を一括して「NN-based surface-code decoder」と呼ぶことはできるが、実態はかなり異なる。
- 最大の分岐は `single-shot vs repeated-rounds`, `physical recovery vs logical observable prediction`, `synthetic rotated planar vs Sycamore experimental path` の 3 軸にある。
- したがって今後の比較や再利用では、まず `ノイズモデル`, `出力の意味`, `surface-code backend` の 3 点で repo を分けてから見るのが妥当である。
