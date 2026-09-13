# python-robot-path-planning

Zenn本『**Pythonではじめるロボット経路生成入門 ─ ダイクストラ法・A\*からRRT・Informed RRT\*まで**』のサンプルコードです。

ロボットの経路生成（パスプランニング：障害物をよけて目的地まで進む道を計算すること）を、
純Python（NumPy / Matplotlib）で1手法ずつ実装し、図と gif で動きを確かめていく教材コードです。
前半は2次元グリッド上の点ロボットで **ダイクストラ法・A\*・ポテンシャル法** を、
後半はサンプリングベースの **PRM・RRT・RRT-Connect・RRT\*・Informed RRT\*** を実装し、
最後に8手法すべてを **2軸ロボットアームの関節空間** に適用して動作を見比べます。

## 動作環境

| ソフトウェア | バージョン | 用途 |
| --- | --- | --- |
| Python | 3.10.9 | 本体 |
| NumPy | 1.26.4 | 数値計算 |
| Matplotlib | 3.7.0 | グラフ・アニメーション（gif）作成 |
| python-fcl | 0.7.0.8 | 干渉判定（第11章のみ使用） |

```bash
pip install numpy matplotlib
pip install python-fcl       # 第11章を動かすときだけ
```

第2〜10章（点ロボット）は NumPy と Matplotlib だけで動きます。衝突判定も外部ライブラリに頼らず
自前の計算で済ませているので、`python-fcl` が必要になるのは現実のアームを扱う第11章だけです。
Windows や Linux でも、同じコードでそのまま動きます。

## 実行方法

各 `partN_xxx/` ディレクトリは**自己完結した実行単位**で、`from constant import *` のように
同じフォルダ内のモジュールをフラットに読み込みます。パッケージ化していないため、
**必ず対象ディレクトリへ `cd` してから実行**してください。

```bash
cd part3_dijkstra
python main.py
```

実行すると、その章の図（`.png`）やアニメーション（`.gif`）が、同じディレクトリに生成されます
（生成物はリポジトリには含めていません。手元で実行して確認してください）。

> `part9_rrt_star/exp_goal_bias.py` は、第9章b の「ゴールバイアスを途中で切り替えると効くのか」を
> シード20本×5設定で実測した検証スクリプトです（本文の表のもと）。画像は書き出さず、数分かかります。

## 章とディレクトリの対応

章番号とディレクトリの番号は一致しています（`main.py` を実行したときの生成物の数）。

| 章 | テーマ | ディレクトリ |
| --- | --- | --- |
| 第2章 | 準備（占有格子地図・近傍・衝突判定・可視化の土台） | `part2_setup` |
| 第3章 | ダイクストラ法（グリッド上の最短経路） | `part3_dijkstra` |
| 第4章 | A\*（ヒューリスティックで展開を絞る。BFS / Greedy との4手法比較つき） | `part4_astar` |
| 第5章 | ポテンシャル法（引力と斥力・局所最小値での失敗例） | `part5_potential_field` |
| 第6章 | PRM（ロードマップを作って A\* で探索） | `part6_prm` |
| 第7章 | RRT（木を伸ばす・ゴールバイアス） | `part7_rrt` |
| 第8章 | RRT-Connect（2本の木を両側から伸ばす） | `part8_rrt_connect` |
| 第9章 | RRT\*（choose parent と rewire で経路を短くする） | `part9_rrt_star` |
| 第10章 | Informed RRT\*（初期解のあと楕円にサンプリングを絞る） | `part10_informed_rrt_star` |
| 第11章 | 2軸アームへの応用（8手法を関節空間で見比べる・python-fcl） | `part11_arm_application` |

## モジュール構成

第2章で作った共通モジュールを、第3章以降の各ディレクトリへ**コピーして持ち回っています**
（1つのディレクトリだけで実行できることを優先したためです）。

| モジュール | 役割 |
| --- | --- |
| `constant.py` | 全モジュール共有の定数（地図サイズ、始点・終点、各手法のパラメータなど） |
| `grid_map.py` | 占有格子地図（点ロボットが動く2次元グリッド）と本書共通のサンプル地図 |
| `collision.py` | 点ロボット用の衝突判定（点・線分。NumPy だけで完結） |
| `plot.py` | 地図・経路・探索の様子を描く共通部品 |
| `main.py` | エントリポイント（`if __name__ == "__main__": main()`） |
| `gen_*.py` | 理論編の概念図を描くスクリプト |

章ごとの手法は、その章の名前を持つモジュールに実装しています。前の章の手法を土台にする章は、
その章のファイルも同梱しています（例：`part9_rrt_star` には `rrt.py` も入っている）。

| モジュール | 初出 | 役割 |
| --- | --- | --- |
| `dijkstra.py` | 第3章 | ダイクストラ法 |
| `astar.py` `search_compare.py` | 第4章 | A\*、および BFS / ダイクストラ法 / Greedy Best-First / A\* の比較 |
| `potential_field.py` | 第5章 | ポテンシャル法 |
| `prm.py` | 第6章 | PRM |
| `rrt.py` | 第7章 | RRT（`Tree` クラス・sample / steer） |
| `rrt_connect.py` | 第8章 | RRT-Connect（extend / connect） |
| `rrt_star.py` | 第9章 | RRT\*（choose parent / rewire） |
| `informed_rrt_star.py` | 第10章 | Informed RRT\*（楕円サンプリング） |
| `robot.py` `rotation.py` `environment.py` `animation.py` | 第11章 | 2軸アームの運動学、回転行列、障害物のある環境、アームの動作 gif |

第11章の `collision.py` だけは、点ロボット用ではなく **2軸アームの干渉判定**に差し替わっています
（アームと障害物を python-fcl の図形として持つのは `robot.py` と `environment.py` です）。
判定関数の名前と呼び出し方は変えていないので、8手法のコードはそのまま動きます。

## 関連

- 📙 Zenn本『Pythonではじめるロボット経路生成入門』（本リポジトリの解説）
- 📕 第1弾『Pythonではじめる2軸ロボットアーム入門』（[python-2dof-robot-arm](https://github.com/haruhiro1020/python-2dof-robot-arm)）
- 📘 第2弾『Pythonではじめる6軸ロボットアーム入門』（[python-6dof-robot-arm](https://github.com/haruhiro1020/python-6dof-robot-arm)）
