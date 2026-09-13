# メイン処理（第2章：地図・近傍・衝突判定・可視化の土台を確認する）

# 自作モジュールの読み込み
from constant import *                  # 定数
from grid_map import make_sample_map    # サンプル地図
from collision import is_collision_segment   # 線分の衝突判定
from plot import plot_map, plot_result  # 可視化


def main() -> None:
    """
    メイン処理
    """
    # 本書共通のサンプル地図を作成
    grid_map = make_sample_map()
    print(f"地図サイズ (行, 列) = {grid_map.shape}")
    print(f"スタート = {grid_map.start}, ゴール = {grid_map.goal}")

    # ① 地図（占有格子）だけを描く
    plot_map(grid_map, "grid_map_basic.png", title="sample grid map")

    # ② 近傍（4近傍 / 8近傍）の動きを確認する
    #    スタートのすぐ上のセルで，進める方向の数を比べる
    sample_cell = (17, 1)
    n4 = grid_map.neighbors(sample_cell, CONNECTIVITY.FOUR)
    n8 = grid_map.neighbors(sample_cell, CONNECTIVITY.EIGHT)
    print(f"セル {sample_cell} の 4近傍 = {len(n4)} 方向, 8近傍 = {len(n8)} 方向")

    # ③ 線分の衝突判定を確認する（壁をまたぐ線分は衝突あり）
    free_segment = is_collision_segment(grid_map,
                                        grid_map.cell_to_pos((18, 1)),
                                        grid_map.cell_to_pos((18, 8)))
    wall_segment = is_collision_segment(grid_map,
                                        grid_map.cell_to_pos((5, 1)),
                                        grid_map.cell_to_pos((5, 18)))
    print(f"床に沿う線分の衝突 = {free_segment}（False を期待）")
    print(f"壁をまたぐ線分の衝突 = {wall_segment}（True を期待）")

    # ④ 手書きの経路（点の列）を地図に重ねて描く
    hand_path = [(18, 1), (12, 5), (8, 10), (4, 14), (1, 18)]
    plot_result(grid_map, hand_path, [], "plot_map_path.png",
                title="map with a hand-drawn path")


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
