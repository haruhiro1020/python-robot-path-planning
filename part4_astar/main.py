# メイン処理（第4章：A* でグリッド上の最短経路を引き，ダイクストラ法と展開ノード数を比べる）

# 自作モジュールの読み込み
from constant import *                  # 定数
from grid_map import make_sample_map    # サンプル地図
from dijkstra import dijkstra           # ダイクストラ法（比較用）
from astar import astar                 # A* 法
from search_compare import bfs, priority_search    # 4手法比較用
from plot import plot_result, plot_compare, plot_search_animation    # 可視化


def main() -> None:
    """
    メイン処理
    """
    # 本書共通のサンプル地図を作成（第3章と同じ地図なので展開ノード数をそのまま比較できる）
    grid_map = make_sample_map()

    # --- A* 法で最短経路を求める（8近傍・ユークリッド距離） ---
    a_path, a_explored, a_cost = astar(grid_map, CONNECTIVITY.EIGHT, HEURISTIC.EUCLIDEAN)
    if not a_path:
        print("経路が見つかりませんでした")
        return

    # --- ダイクストラ法（比較用） ---
    d_path, d_explored, d_cost = dijkstra(grid_map, CONNECTIVITY.EIGHT)

    # 結果を表示（A* はダイクストラ法と同じ経路コストを，より少ない展開で見つけるのが見どころ）
    print("=== Dijkstra vs A* ===")
    print(f"Dijkstra : 通過点数={len(d_path):3d} / 経路コスト={d_cost:.2f} / 展開ノード数={len(d_explored)}")
    print(f"A*       : 通過点数={len(a_path):3d} / 経路コスト={a_cost:.2f} / 展開ノード数={len(a_explored)}")

    # 静止画：A* の最短経路＋探索済みセル
    plot_result(grid_map, a_path, a_explored, "astar_path.png",
                title=f"A* (explored={len(a_explored)}, cost={a_cost:.2f})")

    # 比較図：同一地図でのダイクストラ法と A* の探索範囲（A* がゴール方向へ絞られる）
    plot_compare(grid_map,
                 [("Dijkstra", d_path, d_explored), ("A*", a_path, a_explored)],
                 "astar_vs_dijkstra.png",
                 suptitle="Dijkstra vs A* (same map / same cost)")

    # 比較図（本章の目玉）：BFS / ダイクストラ法 / Greedy / A* を g と h の役割で見比べる
    b_path, b_explored, _ = bfs(grid_map, CONNECTIVITY.EIGHT)
    # priority_search は重みで手法を切り替える（g_weight, h_weight）
    dj_path, dj_explored, _ = priority_search(grid_map, 1.0, 0.0)   # ダイクストラ法（g のみ）
    gr_path, gr_explored, _ = priority_search(grid_map, 0.0, 1.0)   # Greedy（h のみ）
    as_path, as_explored, _ = priority_search(grid_map, 1.0, 1.0)   # A*（g + h）
    plot_compare(grid_map,
                 [("BFS (no cost)", b_path,  b_explored),
                  ("Dijkstra (g)",  dj_path, dj_explored),
                  ("Greedy (h)",    gr_path, gr_explored),
                  ("A* (g+h)",      as_path, as_explored)],
                 "search_compare_bfs_dijkstra_greedy_astar.png",
                 suptitle="BFS / Dijkstra / Greedy / A* : the roles of g and h")

    # アニメーション：ゴール方向へ伸びていく A* の探索の様子
    plot_search_animation(grid_map, a_path, a_explored, "astar_anime.gif",
                          step=10, title="A*")


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
