# メイン処理（第9章：RRT* で choose parent と rewire を足し，反復のたびに経路を短くする）

# ライブラリの読み込み
import numpy as np   # 数値計算（経路長の計算に使う）

# 自作モジュールの読み込み
from grid_map import make_sample_map     # 共通サンプル地図
from rrt import plan_rrt                 # 第7章 RRT（経路コストの比較相手）
from rrt_star import plan_rrt_star       # RRT* 経路生成
from plot import (plot_rrt_tree, plot_rrt_path,
                  plot_cost_curve, animate_rrt_star)   # 可視化


def _path_length(path: list) -> float:
    """
    経路（連続座標のリスト）の総延長を求める（経路の質をコストで比べるため）

    パラメータ
        path: 経路（連続座標 (x, y) のリスト）

    戻り値
        length: 経路の総延長（空なら 0.0）
    """
    if not path:
        return 0.0
    p = np.array(path)
    return float(np.linalg.norm(p[1:] - p[:-1], axis=1).sum())


def main() -> None:
    """
    メイン処理
    """
    # 共通サンプル地図（壁・部屋・狭い通路を含む 20×20）で RRT* を実行
    grid_map = make_sample_map()

    # RRT* で経路生成（choose parent → rewire を反復し，ゴール到達後もコストを下げ続ける）
    path, tree, cost_history, snapshots = plan_rrt_star(grid_map)

    print("=== RRT*（choose parent と rewire で経路を最適へ近づける経路生成）===")
    print(f"木のノード数 = {tree.n_nodes} / 最良経路の更新回数 = {len(snapshots)}")
    if path:
        first_cost = snapshots[0][1]
        last_cost  = snapshots[-1][1]
        print(f"到達 = True / 経路の通過点数 = {len(path)}")
        print(f"初期解コスト = {first_cost:.2f} → 最終コスト = {last_cost:.2f}"
              f"（{first_cost - last_cost:.2f} 短縮）")
    else:
        print("到達 = False（最大試行回数までゴールへ届かなかった）")

    # 図（理論編）：rewire で枝が整理された最終的な木
    plot_rrt_tree(grid_map, tree, "rrt_star_tree.png",
                  title="RRT* tree (rewired)")

    # 図（実装編）：木（薄め）の上に最適化された最良経路
    plot_rrt_path(grid_map, tree, path, "rrt_star_path.png",
                  title="RRT* path (optimized)")

    # 図（実装編）：反復回数 vs 経路コストの低下グラフ
    plot_cost_curve(cost_history, "rrt_star_cost_curve.png",
                    title="RRT* cost vs iteration")

    # gif（実装編）：最良経路が反復のたびに少しずつ短くなるアニメ
    animate_rrt_star(grid_map, tree, snapshots, "rrt_star_anime.gif",
                     title="RRT*")

    # --- 第7章 RRT との経路コスト比較（同じ地図で，経路がどれだけ短くなったか）---
    # ※RRT はゴールに届いた時点で打ち切るので経路が冗長。RRT* は反復で短くする
    print("\n=== RRT と RRT* の経路コスト比較（同じ地図・同じシード）===")
    rrt_path, _, _ = plan_rrt(grid_map)
    rrt_cost  = _path_length(rrt_path)
    star_cost = _path_length(path)
    print(f"{'手法':<10}{'経路コスト':>12}")
    print(f"{'RRT':<12}{rrt_cost:>10.2f}")
    print(f"{'RRT*':<12}{star_cost:>10.2f}")


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
