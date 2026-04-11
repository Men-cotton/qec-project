# NN-based 各 repo の matching 観点での統合メモ

このメモは、更新された各 survey markdown を横断して、
`シンドロームグラフからマッチングを出力する過程` に限って整理し直したものである。

## 対象にした survey markdown

- `neural_network_decoder/repo_investigation.md`
- `astra/repo_surface_code_investigation.md`
- `graphqec-paper/repo_surface_code_investigation.md`
- `QEC_GNN-RNN/repo_surface_code_investigation.md`
- `GNN_decoder/repo_investigation_surface_code.md`
- `VNA-Decoder/surface_code_repo_investigation.md`
- `DeepNeuralDecoder/SURFACE_CODE_REPO_INVESTIGATION.md`

注:
- ここでの要約は、各 repo の raw code を新たに再解釈したものではなく、上記 survey の結論を横断比較しやすい形に並べ直したものである。
- 「対応/未対応」は `surface code を扱うか` ではなく、`matching graph を保持・構築・解くか` を基準にしている。

## 全体結論

- 7 repo のうち、`matching を実装またはラップしている` と言えるのは実質 3 つだけである。
  - `astra`
  - `graphqec-paper`
  - `QEC_GNN-RNN`
- ただし 3 つとも、matching の中核は repo 自前実装ではなく外部 backend 依存である。
  - `astra`: `panqec.decoders.MatchingDecoder` 経由の PyMatching
  - `graphqec-paper`: `PyMatching` wrapper と `BPOSD` 系
  - `QEC_GNN-RNN`: `pymatching.Matching.from_detector_error_model(...)`
- 残り 4 repo は surface code decoder ではあるが、matching repo ではない。
  - `neural_network_decoder`: sequence model による parity prediction
  - `GNN_decoder`: 3D defect kNN graph を使う graph classification
  - `VNA-Decoder`: EWD / MCMC / VNA-RNN
  - `DeepNeuralDecoder`: lookup / pure correction + NN logical stage

## repo 横断比較

| repo | matching 実装/ラップ | 対象グラフ | 2D/3D | 境界ノード | 重み | hyperedge / 相関誤り | repo 境界で見える出力 | NN の役割 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `neural_network_decoder` | なし | 明示的 graph なし。SQLite 由来の syndrome increment tensor | 判定不能。graph 表現なし | なし | なし | なし | logical parity | LSTM で parity 直接予測 |
| `astra` | あり。外部 decoder をラップ | local では Tanner graph、matching 自体は PanQEC/PyMatching 内部 graph | 2D | local 実装なし。外部 decoder 側 | あり。`fllrx/fllrz` を渡す | なし。X/Z 分離 | physical recovery を得て residual error 評価 | GNN は qubit-wise likelihood 推定 |
| `graphqec-paper` | あり。`PyMatching`, `BPOSD`, `SlidingWindowBPOSD` | `stim.DetectorErrorModel` とその incidence matrix | 2D/3D | repo では明示 node なし。DEM / backend 側 | あり。DEM probability / `priors` | あり。DEM 列を保持 | logical observable | NN は別系統。matching edge は出さない |
| `QEC_GNN-RNN` | あり。PyMatching baseline | Stim が作る detector error model | 3D repeated rounds | repo では明示 node なし。backend 側 | あり。ただし外部側 | なし。`decompose_errors=True` | logical observable | GNN+GRU は別経路で logical flip 直接予測 |
| `GNN_decoder` | なし | defect の kNN graph | 3D | なし | あり。幾何距離 `1 / dist^power` | なし | logical observable flip | GNN で graph-level 二値分類 |
| `VNA-Decoder` | 実動コードとしてはなし | 2D syndrome 配列のみ。graph object なし | 2D | half plaquette はあるが boundary node なし | edge weight なし。chain 全体の effective length を使う | なし | 等価類分布 | RNN は correction pattern sampler |
| `DeepNeuralDecoder` | なし | graph なし。`G`, `L`, `T`, `correctionMat` の行列表現 | graph としては未対応 | なし | なし | なし。X/Z へ射影 | logical frame bit と内部 correction state | NN は logical stage の分類器 |

## まず押さえるべき分岐

### 1. matching repo かどうか

- `matching あり`
  - `astra`
  - `graphqec-paper`
  - `QEC_GNN-RNN`
- `matching なし`
  - `neural_network_decoder`
  - `GNN_decoder`
  - `VNA-Decoder`
  - `DeepNeuralDecoder`

この分岐が最重要である。後者 4 つは「surface code を decode する」こと自体はしていても、PLANS.md の意味での `matching graph solver` ではない。

### 2. 2D code-capacity か 3D repeated-round か

- `2D 寄り`
  - `astra`
  - `VNA-Decoder`
- `3D repeated-round / detector-history 寄り`
  - `graphqec-paper`
  - `QEC_GNN-RNN`
  - `GNN_decoder`
  - `neural_network_decoder` は graph は無いが repeated-cycle 系 dataset を扱う
- `中間的`
  - `DeepNeuralDecoder`
  - generator 側は circuit-level / repeated rounds を含むが、decoder 側は 3D graph へは落とさず代表 syndrome 選択へ圧縮する

### 3. hyperedge を保持するか、X/Z 独立 graph に落とすか

- `hyperedge / multi-detector error event を最も素直に保持する`
  - `graphqec-paper`
  - DEM の列をそのまま持ち、`simplify_dem` も same hyper-edge をまとめる
- `X/Z 独立または graph-like へ分解`
  - `astra`
  - `QEC_GNN-RNN`
  - `DeepNeuralDecoder`
- `そもそも matching graph を持たない`
  - `neural_network_decoder`
  - `GNN_decoder`
  - `VNA-Decoder`

## repo ごとの位置づけ

### A. 外部 matching backend を薄く呼ぶ repo

- `QEC_GNN-RNN`
  - 最も単純な matching wrapper である。
  - `Stim` で回路と DEM を作り、`PyMatching` にそのまま流す。
  - repo 内で見える入出力は `detector event bool array -> logical observable prediction` までで、matching pair や edge list は露出しない。

- `astra`
  - local に作っている graph は Tanner graph であり、matching graph ではない。
  - matching は GNN の後段で `MatchingDecoder` を都度呼ぶ 2 段構成である。
  - NN が直接出すのは matching edge ではなく qubit-wise likelihood であり、それを `weights=(fllrx, fllrz)` として外部 decoder に渡す。

### B. detector-error-model を中核にした repo

- `graphqec-paper`
  - 7 repo 中で最も `matching` という観点に近い。
  - `stim.DetectorErrorModel` を中間表現として明示的に使い、`PyMatching`, `BPOSD`, `SlidingWindowBPOSD` を同じ backend 上で切り替える。
  - 2D code-capacity 専用 graph を handcraft するのではなく、`num_cycle` に応じて 2D/3D の detector-history graph 相当を DEM として構成する。
  - ただし public output はここでも matching pair ではなく logical observable prediction である。

### C. graph は使うが matching ではない repo

- `GNN_decoder`
  - repeated rounds の detector event から 3D defect kNN graph を動的生成する。
  - しかしその graph は GNN 入力用であり、matching solver 用ではない。
  - edge weight も `1 / dist^power` という message passing 補助特徴であって matching cost ではない。

- `neural_network_decoder`
  - graph 表現自体を持たず、syndrome increment sequence と final syndrome increment をそのまま LSTM へ入れる。
  - 出力は parity 1 ビットであり、matching との接点はほぼない。

### D. correction pattern / logical class を扱う repo

- `VNA-Decoder`
  - 2D rotated planar patch の syndrome 配列は持つが、graph object を持たない。
  - reference decoder は EWD/MCMC であり、edge 単位の matching cost ではなく chain 全体の effective length を使う。
  - `mwpm_init` の痕跡はあるが、survey 上は実動 matching 実装として扱えない。

- `DeepNeuralDecoder`
  - generator 側には repeated rounds と circuit-level fault があるが、decoder 側はそれを graph にせず代表 syndrome へ縮約する。
  - base decoder は lookup / pure correction であり、MWPM 的な path search は無い。
  - NN は correction graph を解くのではなく logical frame の最終分類に使われる。

## matching 観点で最も差が出る軸

### 1. local repo が graph を持っているか

- `graph を local に持つ`
  - `graphqec-paper`: DEM incidence matrix
  - `GNN_decoder`: defect kNN graph
  - `astra`: Tanner graph
  - `QEC_GNN-RNN`: neural 側では kNN graph
- `graph を local に持たない`
  - `neural_network_decoder`
  - `VNA-Decoder`
  - `DeepNeuralDecoder`

ただし、`local に graph を持つ` ことと `matching graph を解く` ことは別である。`astra` と `GNN_decoder` の local graph は主に NN 用であり、matching 用 graph そのものではない。

### 2. matching backend が repo の外に隠れているか

- `ほぼ完全に外部へ委譲`
  - `astra`
  - `QEC_GNN-RNN`
- `中間表現までは repo 内で明示`
  - `graphqec-paper`
- `backend 自体が無い`
  - 残り 4 repo

### 3. repo 境界で返すものが何か

- `logical observable / parity`
  - `neural_network_decoder`
  - `graphqec-paper`
  - `QEC_GNN-RNN`
  - `GNN_decoder`
  - `DeepNeuralDecoder`
- `physical recovery を中間的に使う`
  - `astra`
- `logical equivalence class 分布`
  - `VNA-Decoder`

このため、matching を使う repo であっても、統合インターフェースとして `matching pair list` を返すものは確認されなかった。

## 実務上の見方

- `matching graph 自体を調べたい` なら、主対象は `graphqec-paper` である。
  - DEM、priors、hyperedge、sliding window まで最も明示的である。

- `PyMatching を使った最小ラッパを見たい` なら、`QEC_GNN-RNN` が最も単純である。
  - `Stim DEM -> PyMatching.decode_batch` という流れが明確である。

- `NN で重みを推定し、matching を後段に置く構成` を見たいなら、`astra` が最も近い。
  - ただし local graph は Tanner graph であり、decoding graph を repo 自身で実装しているわけではない。

- `NN-based surface-code decoder` という名前でも matching を期待してはいけない repo が多い。
  - `neural_network_decoder`
  - `GNN_decoder`
  - `VNA-Decoder`
  - `DeepNeuralDecoder`

## 結論

- この commit 時点の NN-based survey を matching 観点で見直すと、全 repo を一括して `NN-based matching decoder` と呼ぶのは不正確である。
- 実際には次の 3 類型に分かれる。
  - `外部 matching backend を呼ぶ repo`: `astra`, `graphqec-paper`, `QEC_GNN-RNN`
  - `graph を使うが matching しない repo`: `GNN_decoder`
  - `graph 自体を持たず parity / class / logical frame を直接推定する repo`: `neural_network_decoder`, `VNA-Decoder`, `DeepNeuralDecoder`
- matching の実装密度が最も高いのは `graphqec-paper`、最も薄い wrapper は `QEC_GNN-RNN`、NN と matching の二段構成が最も明確なのは `astra` である。
