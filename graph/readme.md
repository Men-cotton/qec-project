# `graph/` JSON Data Notes

このディレクトリには、Stim 由来の detector 配置から構成したグラフ JSON が入っている。
各ファイルは 1 個のサンプルを表し、距離 `d`、測定ラウンド数 `r`、case index を
ファイル名に持つ。

例:

- `graph_data_d5_r5_case_0.json`
- `graph_data_d9_r9_case_17.json`

## JSON に入っているもの

各 JSON には主に次が入っている。

- 基本メタデータ
  - `code_distance`
  - `measurement_rounds`
- 3 種類のグラフ表現
  - `short_subgraph`
  - `long_subgraph`
  - `fullgraph`
- 各グラフ表現ごとの頂点情報
  - `*_node_ids`
  - `*_boundary_node_ids`
- 各グラフ表現ごとの MWPM 情報
  - `*_MWPM_weight`
  - `*_MWPM_matching`
  - `*_MWPM_is_valid`
- 追加の補助情報
  - `boundary_node_id`
  - `imperfect_fmu_weight`
  - `imperfect_fmu_matching`
  - `imperfect_fmu_is_valid`

辺は `[vertex_1, vertex_2, weight]` の形で保存される。
MWPM の matching は `[vertex_1, vertex_2]` の形で保存される。

## `short`, `long`, `full` の意味

この repo にある JSON だけを見る限り、3 つは同じ defect 配置に対する
「候補辺の作り方が違うグラフ」と読むのが自然である。

### `short_subgraph`

- 最も疎なグラフ。
- `long_subgraph` と比べて、頂点集合はほぼ同じだが、候補辺が少ない。
- 解釈としては、近距離または局所的な接続候補だけを残した版と考えるのが自然。

### `long_subgraph`

- `short_subgraph` を拡張した中間のグラフ。
- `short_subgraph` と regular node / boundary node の数は一致することが多い。
- 差は主に辺集合にあり、`short` では落としていたより長い候補辺を足した版とみなせる。

### `fullgraph`

- 最も包括的なグラフ。
- 辺数は `short` や `long` より大きく、特に高距離では非常に密になる。
- `short` / `long` と同じ regular node を持ちつつ、boundary node 側の候補も
  より広く持つ版になっている。
- 解釈としては、候補辺の制限をかなり外した「完全版の syndrome graph」に近い。

## 実際の関係

全ケースを集計すると、次の傾向が見える。

- `short` と `long` の違いは主に辺集合である。
- `full` は `long` よりさらに多くの辺を持つ。
- `full` は boundary 側の候補頂点も増えることがある。
- MWPM の最小重みは `short` と `long` で一致することが多く、
  `full` でも大半のケースで一致する。

したがって、読み方としては次でよい。

- `short`: 強く制限した疎グラフ
- `long`: 同じ頂点集合のまま長い候補辺を足した中間グラフ
- `full`: boundary 側も含めて候補を広く持つ dense graph

## 注意

この解釈は、現在の repo にある JSON の構造と集計結果から得たものである。
このディレクトリにはグラフ生成コード自体は入っていないため、
`short` / `long` の厳密な構成規則まではこの README だけでは断定できない。

ただし、少なくとも運用上は、

- `short`, `long`, `full` は同じサンプルの別グラフ表現
- 疎さは `short < long < full`
- MWPM 情報は各グラフ表現ごとに独立に付いている

と理解しておけばよい。

## MWPM 情報をどう使うか

この JSON には、各グラフに対して少なくとも 1 つの MWPM 解と、その重みが保存されている。

ただし MWPM は一般に一意ではない。
同じ最小重みを持つ matching が複数存在し得るため、保存されている matching は
「そのうちの 1 つ」とみなすべきである。

NN 学習で使う候補としては、少なくとも次が考えられる。

- MWPM の最小重み
- 代表となる 1 つの MWPM 解

「全ての MWPM 解」が必要かどうかは、この JSON からは決まらない。

## `imperfect_fmu` について

`imperfect_fmu_*` は、`short` / `long` / `full` の各 `*_MWPM_*` とは別系統の
復号結果として保存されている。

この repo の JSON を全件集計すると、少なくとも次は事実として言える。

- `short_subgraph_MWPM_is_valid`
- `long_subgraph_MWPM_is_valid`
- `fullgraph_MWPM_is_valid`

は全 500 件で常に `true` である。

一方で `imperfect_fmu_is_valid` は大きく変動し、全体では

- `true`: 236 件
- `false`: 264 件

であった。

距離ごとの内訳は次の通り。

- `d5_r5`: valid 100, invalid 0
- `d9_r9`: valid 93, invalid 7
- `d13_r13`: valid 39, invalid 61
- `d17_r17`: valid 3, invalid 97
- `d21_r21`: valid 1, invalid 99

このため、`imperfect_fmu_is_valid` は JSON 構文の妥当性ではなく、
`imperfect_fmu` という別系統の matching 結果が有効に得られたかどうかを
表すフラグとみるのが自然である。

### `is_valid = false` のとき

`imperfect_fmu_is_valid == false` の 264 件すべてで、
`imperfect_fmu_weight` は

- `18446744073709551615`

だった。これは `u64::MAX` 相当の sentinel 値と考えるのが自然で、
通常の重みではなく「失敗」または「未定義」の印として使われている可能性が高い。

例:

- `graph_data_d9_r9_case_65.json`
- `graph_data_d17_r17_case_76.json`

### `is_valid = true` のとき

`imperfect_fmu_is_valid == true` の 236 件について、
`imperfect_fmu_weight` と `fullgraph_MWPM_weight` を比べると、

- `imperfect_fmu_weight < fullgraph_MWPM_weight`: 0 件
- `imperfect_fmu_weight == fullgraph_MWPM_weight`: 122 件
- `imperfect_fmu_weight > fullgraph_MWPM_weight`: 114 件

だった。

したがって、`imperfect_fmu` は少なくとも `fullgraph_MWPM` より良い最適解を
与えるものではなく、

- 同じ重みの解を返すこともある
- ただし高距離ではより重い解になることが多い
- さらに高距離では valid 自体が急減する

という振る舞いを示す。

距離ごとの傾向も明確で、

- `d5_r5`: 100 件中 97 件で `fullgraph_MWPM_weight` と一致
- `d9_r9`: valid 93 件中 68 件で `fullgraph_MWPM_weight` より重い
- `d13_r13`: valid 39 件すべてで `fullgraph_MWPM_weight` より重い
- `d17_r17`: valid 3 件すべてで `fullgraph_MWPM_weight` より重い
- `d21_r21`: valid 1 件で `fullgraph_MWPM_weight` より重い

となっている。

## 論文・発表資料との対応づけ

提供資料中に `imperfect_fmu` という文字列そのものは見当たらなかったが、
資料に出てくる

- `Final Matching Unit (FMU)`
- `Simple-FMU`
- `Greedy FMU`
- 厳密 MWPM を与える `Combinatorial-FMU` との対比

という説明と、この JSON の統計はかなり整合している。

JSON から安全に言えるのは、`imperfect_fmu` が

- 厳密な `fullgraph_MWPM` とは別の手法の出力で
- 近似的または失敗しうる matching 結果であり
- 距離が大きくなると品質と完遂率が悪化する

という点である。

これは、発表資料で述べられている

- 低遅延のために近似的な FMU を使う
- その結果、高距離で性能スケーリングが悪化する

という説明と整合する。

## ただし断定できない点

この repo には `graph/*.json` の生成コードが入っていないため、
JSON だけから次を断定することはできない。

- `imperfect_fmu` が資料中の `Simple-FMU` / `Greedy FMU` と完全に同一の実装であること
- `FMU` がここで厳密に `Final Matching Unit` を意味していること
- `imperfect_fmu_matching` が `fullgraph` の辺をそのまま並べた matching であること

特に最後の点は重要で、`imperfect_fmu_matching` に含まれる node id / pair は
`fullgraph_MWPM_matching` と同じ表現とは限らず、
`fullgraph` に保存された辺集合へそのまま 1 対 1 に落ちるわけではない。

そのため、この README では次の慎重な理解を採る。

- `imperfect_fmu` は exact MWPM とは別系統の近似的 matching 結果
- `imperfect_fmu_is_valid` はその結果が有効に得られたかどうか
- `imperfect_fmu_weight` は valid なときのみ意味のあるコスト
- invalid のときの重み `18446744073709551615` は sentinel とみなす

実運用上は、この理解で扱うのが最も安全である。
