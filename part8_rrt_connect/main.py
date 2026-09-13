# メイン処理（第8章：RRT-Connect で2本の木を両側から伸ばし，中央で接続して経路を引く）

# ライブラリの読み込み
import time   # 実行時間の計測（RRT との速度比較に使う）

# 自作モジュールの読み込み
from grid_map import make_sample_map         # 共通サンプル地図
from rrt import plan_rrt                     # 第7章 RRT（速度比較の相手）
from rrt_connect import plan_rrt_connect     # RRT-Connect 経路生成
from plot import (plot_rrt_connect_trees, plot_rrt_connect_path,
                  animate_rrt_connect)       # 可視化


def _path_length(path: list) -> float:
    """
    経路（連続座標のリスト）の総延長を求める（経路の長さで質を比べるため）

    パラメータ
        path: 経路（連続座標 (x, y) のリスト）

    戻り値
        length: 経路の総延長（空なら 0.0）
    """
    import numpy as np
    if not path:
        return 0.0
    p = np.array(path)
    return float(np.linalg.norm(p[1:] - p[:-1], axis=1).sum())


def main() -> None:
    """
    メイン処理
    """
    # 共通サンプル地図（壁・部屋・狭い通路を含む 20×20）で RRT-Connect を実行
    grid_map = make_sample_map()

    # RRT-Connect で経路生成（A を1歩伸ばす → B を一気に CONNECT → swap を繰り返す）
    start = time.perf_counter()
    path, tree_start, tree_goal, events, n_iterations = plan_rrt_connect(grid_map)
    connect_time = time.perf_counter() - start
    n_nodes = tree_start.n_nodes + tree_goal.n_nodes

    print("=== RRT-Connect（両側から木を伸ばして速く接続する経路生成）===")
    print(f"反復回数 = {n_iterations} / 2木の合計ノード数 = {n_nodes}")
    if path:
        print(f"到達 = True / 経路の通過点数 = {len(path)} / 経路コスト = {_path_length(path):.2f}")
    else:
        print("到達 = False（最大反復回数まで2本の木が出会わなかった）")

    # 図（理論編）：両側から広がり中央付近で出会う2本の木（スタート木＝青／ゴール木＝紫）だけを描く
    # （復元経路は次の rrt_connect_path.png で重ねるので，ここでは木の広がりだけを見せる）
    plot_rrt_connect_trees(grid_map, tree_start, tree_goal,
                           "rrt_connect_two_trees.png",
                           title="RRT-Connect two trees (start vs goal)")

    # 図（実装編）：2本の木（薄め）の上に結合・復元した経路
    plot_rrt_connect_path(grid_map, tree_start, tree_goal, path,
                          "rrt_connect_path.png",
                          title="RRT-Connect path (two trees + merged path)")

    # gif（実装編）：両側から枝を交互に伸ばし → 出会って経路を引くアニメ
    animate_rrt_connect(grid_map, events, path,
                        "rrt_connect_anime.gif", title="RRT-Connect")

    # --- 第7章 RRT との速度比較（同じ地図で，乱数シードを変えて何度も解いた平均で比べる）---
    # ※乱数を使う手法は1回の結果がぶれるので，複数シードの平均で「速さの傾向」を見る
    print("\n=== RRT と RRT-Connect の比較（同じ地図・複数シードの平均）===")
    compare_rrt_and_connect(grid_map, n_trials=20)


def compare_rrt_and_connect(grid_map, n_trials: int = 20) -> None:
    """
    RRT と RRT-Connect を同じ地図で n_trials 回（シードを変えて）解き，平均で比較する

    1回の結果は乱数でぶれるため，複数シードの平均をとって「反復回数・ノード数・時間」の
    傾向を見比べる（RRT-Connect が両側探索で平均的に速く・小さく解けることを確かめる）。

    パラメータ
        grid_map: 占有格子地図
        n_trials: 試行回数（変えるシードの数）
    """
    rrt_iters, rrt_nodes, rrt_times = [], [], []
    rc_iters,  rc_nodes,  rc_times  = [], [], []

    for seed in range(n_trials):
        # RRT（単一の木）を seed で実行
        t0 = time.perf_counter()
        _, rrt_tree, rrt_n = plan_rrt(grid_map, seed=seed)
        rrt_times.append(time.perf_counter() - t0)
        rrt_iters.append(rrt_n)
        rrt_nodes.append(rrt_tree.n_nodes)

        # RRT-Connect（2本の木）を同じ seed で実行
        t0 = time.perf_counter()
        _, ts, tg, _, rc_n = plan_rrt_connect(grid_map, seed=seed)
        rc_times.append(time.perf_counter() - t0)
        rc_iters.append(rc_n)
        rc_nodes.append(ts.n_nodes + tg.n_nodes)

    def mean(values):
        return sum(values) / len(values)

    print(f"（{n_trials} シードの平均）")
    print(f"{'手法':<14}{'反復回数':>9}{'ノード数':>10}{'時間[ms]':>11}")
    print(f"{'RRT':<16}{mean(rrt_iters):>9.1f}{mean(rrt_nodes):>10.1f}"
          f"{mean(rrt_times) * 1000:>11.1f}")
    print(f"{'RRT-Connect':<14}{mean(rc_iters):>9.1f}{mean(rc_nodes):>10.1f}"
          f"{mean(rc_times) * 1000:>11.1f}")


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
