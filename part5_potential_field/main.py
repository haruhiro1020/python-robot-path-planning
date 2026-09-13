# メイン処理（第5章：ポテンシャル法で連続空間の経路を作り，成功例と失敗例〔局所最小値〕を見る）

# ライブラリの読み込み
import numpy as np
from functools import partial    # 関数の引数を一部固定して別の関数を作る

# 自作モジュールの読み込み
from constant import *               # 定数
from grid_map import GridMap         # 占有格子地図
from potential_field import (        # ポテンシャル法
    obstacle_positions, attractive_force, repulsive_force,
    total_force, plan_potential_field,
)
from plot import (                   # 可視化
    plot_vector_field, plot_potential_path, animate_potential_path,
)


def make_success_map() -> GridMap:
    """
    ポテンシャル法が「ゴールへ到達できる」サンプル地図を作成

    横壁を1本だけ置き，端に隙間を残す。スタートとゴールをずらして配置するので，
    ロボットは斥力に押されながら壁の端を回り込み，ゴールへたどり着ける。

    戻り値
        grid_map: 占有格子地図（スタート・ゴール設定済み）
    """
    rows, cols = 20, 20
    grid = np.zeros((rows, cols), dtype=int)

    # 横壁（左寄り）。右側を大きく空けておき，そこを通って回り込めるようにする
    grid[10, 0:12] = CELL_OBSTACLE

    # スタート（左下）とゴール（右上）。壁の右の隙間を通る想定
    start = (17, 3)
    goal  = (2, 16)
    return GridMap(grid, start, goal)


def make_local_minimum_map() -> GridMap:
    """
    ポテンシャル法が「局所最小値に捕まって失敗する」サンプル地図を作成

    横壁を中央に長く置き，スタートを壁の真下中央・ゴールを真上中央に置く。
    引力（上向き）と，左右対称な壁からの斥力（下向き）が中央で釣り合い，
    ロボットは壁の手前で止まってしまう（＝局所最小値）。

    戻り値
        grid_map: 占有格子地図（スタート・ゴール設定済み）
    """
    rows, cols = 20, 20
    grid = np.zeros((rows, cols), dtype=int)

    # 横壁（中央に長く）。左右対称なので中央で斥力の横成分が打ち消し合う
    grid[10, 2:18] = CELL_OBSTACLE

    # スタート（壁の真下中央）とゴール（壁の真上中央）
    start = (17, 10)
    goal  = (2, 10)
    return GridMap(grid, start, goal)


def main() -> None:
    """
    メイン処理
    """
    # ============ 1. 成功例：壁の端を回り込んでゴールへ ============
    success_map = make_success_map()
    obstacles   = obstacle_positions(success_map)
    goal_pos    = success_map.cell_to_pos(success_map.goal)

    # 各種の力を「位置だけの関数」にして可視化へ渡す（quiver 用）
    f_att   = partial(attractive_force, goal=goal_pos)
    f_rep   = partial(repulsive_force,  obstacles=obstacles)
    f_total = partial(total_force, goal=goal_pos, obstacles=obstacles)

    # 図（理論編）：引力だけのベクトル場（どこにいてもゴールを向く）
    plot_vector_field(success_map, f_att, "potential_attractive.png",
                      title="Attractive field (toward goal)")
    # 図（理論編）：斥力だけのベクトル場（障害物まわりだけ押し返す）
    plot_vector_field(success_map, f_rep, "potential_repulsive.png",
                      title="Repulsive field (around obstacles)")

    # ポテンシャル法で経路生成
    path, reached = plan_potential_field(success_map)
    print("=== ポテンシャル法：成功例 ===")
    print(f"到達 = {reached} / ステップ数 = {len(path)}")

    # 図（実装編）：合力の場＋成功経路
    plot_potential_path(success_map, path, "potential_field_path.png",
                        reached=reached, force_func=f_total,
                        title="Potential field path (success)")
    # gif（実装編）：力に沿って進むアニメ
    animate_potential_path(success_map, path, "potential_field_anime.gif",
                           title="Potential field")

    # ============ 2. 失敗例：壁の手前で局所最小値に捕まる ============
    fail_map       = make_local_minimum_map()
    fail_obstacles = obstacle_positions(fail_map)
    fail_goal      = fail_map.cell_to_pos(fail_map.goal)
    fail_total     = partial(total_force, goal=fail_goal, obstacles=fail_obstacles)

    fail_path, fail_reached = plan_potential_field(fail_map)
    stop = fail_path[-1]
    print("=== ポテンシャル法：失敗例（局所最小値）===")
    print(f"到達 = {fail_reached} / ステップ数 = {len(fail_path)} / 停止位置 = ({stop[0]:.1f}, {stop[1]:.1f})")

    # 図（理論編）：合力の場（中央で引力と斥力が打ち消し合う様子）
    plot_vector_field(fail_map, fail_total, "potential_local_minimum.png",
                      title="Local minimum (forces cancel out)")
    # 図（実装編）：壁の手前で止まる失敗経路
    plot_potential_path(fail_map, fail_path, "potential_field_fail.png",
                        reached=fail_reached, force_func=fail_total,
                        title="Potential field path (stuck at local minimum)")


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
