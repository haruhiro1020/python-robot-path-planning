# メイン処理（第11章：8手法すべてを2軸ロボットアームの関節空間へ適用して見比べる）
#
# これまで点ロボットで学んだ8手法を，同じ2軸アーム・同じ障害物配置・同じ始点終点に対して
# 関節空間（C-space）で動かします。流れは次のとおり：
#   ① 関節空間を格子に区切った C-space地図を作る（禁止領域は python-fcl で判定）
#   ② グラフ系（ダイクストラ法/A*）は離散化した格子を，サンプリング系（PRM/RRT/…）は連続的な
#      関節空間を探索する。手法ごとに「経路・探索の広がり・到達可否・経路コスト・計算時間」を集める
#   ③ 8手法を1枚に並べた C-space比較図を作る
#   ④ 各手法について，アームが障害物をよけて動く gif を作る
#   ⑤ 比較表を出力する

# ライブラリの読み込み
import time             # 計算時間の測定
import numpy as np      # 数値計算

# 自作モジュールの読み込み
from constant import *                          # 定数
from grid_map import make_cspace_map            # C-space地図づくり
from dijkstra import dijkstra                   # ダイクストラ法
from astar import astar                         # A*法
from potential_field import plan_potential_field    # ポテンシャル法
from prm import plan_prm                        # PRM
from rrt import plan_rrt                        # RRT
from rrt_connect import plan_rrt_connect        # RRT-Connect
from rrt_star import plan_rrt_star              # RRT*
from informed_rrt_star import plan_informed_rrt_star   # Informed RRT*
from plot import plot_cspace_compare, plot_workspace_to_cspace   # C-space の可視化
from animation import ArmAnimation              # アーム動作の gif


# 比較図で木・ロードマップの枝を描きすぎないよう，描く枝数の上限（多いと図がつぶれるため）
_MAX_EDGES_TO_DRAW = 1500


def _path_length(path: list) -> float:
    """
    経路（連続座標の列）の全長を測る（手法間でコストをそろえて比べるため）

    パラメータ
        path: 連続座標 (x, y) のリスト

    戻り値
        length: 経路の全長（隣り合う点のユークリッド距離の和）
    """
    if len(path) < 2:
        return 0.0
    points = np.array(path, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _cells_to_coords(cspace_map, cells: list) -> list:
    """
    セル (row, col) のリストを連続座標 (x, y) のリストへ変換する（グラフ系の経路・探索済みセル用）
    """
    return [cspace_map.cell_to_pos(cell) for cell in cells]


def _run_all_methods(cspace_map) -> list:
    """
    8手法を順に実行し，比較用に結果をそろえた辞書のリストを返す

    各辞書は name（手法名）/ path（経路の座標列）/ edges（木・ロードマップの枝）/
    explored（探索済みセル）/ reached（到達可否）/ cost（経路コスト）/ nodes（探索規模）/
    time（計算時間）を持つ。

    パラメータ
        cspace_map: 関節空間（C-space）地図

    戻り値
        results: 各手法の結果辞書のリスト
    """
    results = []

    # --- グラフ系：離散化した格子を探索する ---
    # ダイクストラ法
    t0 = time.time()
    path_cells, explored_cells, cost = dijkstra(cspace_map, CONNECTIVITY.EIGHT)
    results.append({"name": "Dijkstra", "time": time.time() - t0,
                    "path": _cells_to_coords(cspace_map, path_cells),
                    "explored": _cells_to_coords(cspace_map, explored_cells),
                    "edges": None, "reached": len(path_cells) > 0,
                    "cost": cost, "nodes": len(explored_cells)})

    # A*
    t0 = time.time()
    path_cells, explored_cells, cost = astar(cspace_map, CONNECTIVITY.EIGHT, HEURISTIC.EUCLIDEAN)
    results.append({"name": "A*", "time": time.time() - t0,
                    "path": _cells_to_coords(cspace_map, path_cells),
                    "explored": _cells_to_coords(cspace_map, explored_cells),
                    "edges": None, "reached": len(path_cells) > 0,
                    "cost": cost, "nodes": len(explored_cells)})

    # --- ポテンシャル法：連続的な関節空間を合力に沿って進む（局所最小値に注意）---
    t0 = time.time()
    path, reached = plan_potential_field(cspace_map)
    results.append({"name": "Potential Field", "time": time.time() - t0,
                    "path": path, "explored": None, "edges": None,
                    "reached": reached, "cost": _path_length(path), "nodes": len(path)})

    # --- サンプリング系：連続的な関節空間をサンプリングで探索する ---
    # PRM
    t0 = time.time()
    path, roadmap, cost = plan_prm(cspace_map)
    edges = roadmap.edge_list()[:_MAX_EDGES_TO_DRAW]
    results.append({"name": "PRM", "time": time.time() - t0,
                    "path": path, "explored": None, "edges": edges,
                    "reached": len(path) > 0, "cost": _path_length(path),
                    "nodes": roadmap.n_nodes})

    # RRT
    t0 = time.time()
    path, tree, _n_iter = plan_rrt(cspace_map)
    results.append({"name": "RRT", "time": time.time() - t0,
                    "path": path, "explored": None, "edges": tree.edge_list(),
                    "reached": len(path) > 0, "cost": _path_length(path),
                    "nodes": tree.n_nodes})

    # RRT-Connect
    t0 = time.time()
    path, tree_start, tree_goal, _events, _n_iter = plan_rrt_connect(cspace_map)
    results.append({"name": "RRT-Connect", "time": time.time() - t0,
                    "path": path, "explored": None,
                    "edges": tree_start.edge_list() + tree_goal.edge_list(),
                    "reached": len(path) > 0, "cost": _path_length(path),
                    "nodes": tree_start.n_nodes + tree_goal.n_nodes})

    # RRT*
    t0 = time.time()
    path, tree, _cost_history, _snapshots = plan_rrt_star(cspace_map)
    results.append({"name": "RRT*", "time": time.time() - t0,
                    "path": path, "explored": None, "edges": tree.edge_list(),
                    "reached": len(path) > 0, "cost": _path_length(path),
                    "nodes": tree.n_nodes})

    # Informed RRT*
    t0 = time.time()
    path, tree, _cost_history, _snapshots, _samples = plan_informed_rrt_star(cspace_map)
    results.append({"name": "Informed RRT*", "time": time.time() - t0,
                    "path": path, "explored": None, "edges": tree.edge_list(),
                    "reached": len(path) > 0, "cost": _path_length(path),
                    "nodes": tree.n_nodes})

    return results


def _print_table(results: list) -> None:
    """
    8手法の比較表（到達可否・経路コスト・探索規模・計算時間）を出力する
    """
    print("\n=== 8手法の比較（同じ2軸アーム・同じ障害物・同じ始点終点）===")
    print(f"{'手法':<16}{'到達':<6}{'経路コスト':>10}{'探索規模':>10}{'時間[s]':>10}")
    for r in results:
        reached = "○" if r["reached"] else "×"
        cost = f"{r['cost']:.2f}" if r["reached"] else "-"
        print(f"{r['name']:<16}{reached:<6}{cost:>10}{r['nodes']:>10}{r['time']:>10.2f}")


def _method_to_filename(name: str) -> str:
    """
    手法名を gif のファイル名用の文字列へ変換する（例 "Informed RRT*" -> "informed_rrt_star"）
    """
    return name.lower().replace("*", "_star").replace("-", "_").replace(" ", "_").strip("_")


def main() -> None:
    """
    メイン処理
    """
    # ① C-space地図を作る（関節空間を格子に区切り，禁止領域を python-fcl で判定）
    print("C-space地図を作成中（python-fcl でアームの禁止領域を判定）...")
    cspace_map = make_cspace_map()
    print(f"  C-space地図サイズ = {cspace_map.shape} / 禁止領域の割合 = {cspace_map.grid.mean():.3f}")
    print(f"  スタートセル = {cspace_map.start} / ゴールセル = {cspace_map.goal}")

    # 作業空間 → 関節空間（C-space）の対応図（11a 理論編）
    plot_workspace_to_cspace(cspace_map, "ch11_workspace_to_cspace.png")
    print("  ch11_workspace_to_cspace.png を作成しました")

    # ② 8手法を実行
    print("\n8手法を実行中...")
    results = _run_all_methods(cspace_map)

    # ③ C-space比較図
    plot_cspace_compare(cspace_map, results, "ch11_cspace_compare.png")
    print("ch11_cspace_compare.png を作成しました")

    # ④ 各手法のアーム動作 gif
    print("\nアーム動作の gif を作成中...")
    arm_anime = ArmAnimation()
    for r in results:
        file_name = f"ch11_{_method_to_filename(r['name'])}_arm_anime.gif"
        print(f"  {r['name']} -> {file_name}")
        arm_anime.plot_animation(cspace_map, r["path"], file_name, title=f"{r['name']} on 2-DoF arm")

    # ⑤ 比較表
    _print_table(results)


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
