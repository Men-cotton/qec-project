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

## 頂点 id と boundary 頂点

各 graph の頂点 id は `1..n` の範囲にあるが、元の qubit 問題に由来するため、
欠番があり得る。dense な連番へ再番号付けすることもできるが、現状はその必要はない。

`short_subgraph` と `long_subgraph` では、頂点 id 空間は次の 2 つに分かれる。

- regular vertices: `1` から `boundary_node_id - 1`
- boundary vertices: `boundary_node_id` から `2 * boundary_node_id`

したがって、この 2 つの graph では `vertex_id >= boundary_node_id` なら
boundary 頂点である。

`*_boundary_node_ids` は可能な boundary id 全体ではなく、その run で実際に現れた
boundary 頂点 id の列である。

例:

- `short_subgraph_boundary_node_ids = [66, 79, 81, 74, 80, 65]`

boundary id 自体には boundary の位置情報も含まれている。

ただし `fullgraph` の boundary 頂点は別の列挙規約を使っている。
`short` / `long` の規約をそのまま `fullgraph` に当てはめないこと。

## boundary を含む matching

boundary 頂点が matching に使われた場合、その pair は matching 出力に現れる。
形式は

- `[regular_node_id, boundary_id]`

である。

## `short`, `long`, `full` の意味

3 つは同じサンプルに対する別グラフ表現であり、命名は次の意味で読む。

- `fullgraph`: original full syndrome graph
- `long_subgraph`: reduced graph
- `short_subgraph`: minimal graph

ただし `fullgraph` の boundary 頂点の列挙規約は `short` / `long` と異なる。

運用上は、`short`, `long`, `full` は同じサンプルの別表現として扱えばよい。
各表現ごとに MWPM 情報が独立に付いている。

この README は JSON の使い方の説明であり、各 graph の厳密な生成規則までは説明しない。

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

`imperfect_fmu_*` は内部の FPGA decoder で使っている補助情報であり、
通常の graph 利用では無視してよい。特に `imperfect_fmu_matching` は
downstream の use case には有用ではない。

`imperfect_fmu_matching` に現れる `[0, 0]` は matching pair ではなく、
内部で使う end-of-packet signal である。

- `0` は予約された特別値であり、実際の vertex id にはならない
- 内部では、boundary 頂点を持つ graph を boundary なしのより大きい graph に
  変換する際にも `0` を使う

したがって `imperfect_fmu_matching` は解釈せず、`[0, 0]` は無視してよい。
