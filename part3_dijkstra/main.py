# メイン処理（第3章：ダイクストラ法でグリッド上の最短経路を引く）

# 自作モジュールの読み込み
from constant import *                  # 定数
from grid_map import make_sample_map    # サンプル地図
from dijkstra import dijkstra           # ダイクストラ法
from plot import plot_result, plot_search_animation     # 可視化


def main() -> None:
    """
    メイン処理
    """
    # 本書共通のサンプル地図を作成
    grid_map = make_sample_map()

    # ダイクストラ法で最短経路を求める（8近傍）
    path, explored, cost = dijkstra(grid_map, CONNECTIVITY.EIGHT)

    if not path:
        # ゴールに到達できなかった
        print("経路が見つかりませんでした")
        return

    # 結果を表示（展開ノード数は第4章 A* との比較に使う）
    print(f"通過点数（セル数）= {len(path)}")
    print(f"経路コスト       = {cost:.2f}")
    print(f"展開ノード数     = {len(explored)}")

    # 静止画：最短経路＋探索済みセル
    plot_result(grid_map, path, explored, "dijkstra_path.png",
                title=f"Dijkstra (explored={len(explored)}, cost={cost:.2f})")

    # アニメーション：探索フロントが広がり，最後に経路が引かれる様子
    plot_search_animation(grid_map, path, explored, "dijkstra_anime.gif",
                          step=10, title="Dijkstra")


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
