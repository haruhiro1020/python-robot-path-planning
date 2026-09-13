# 点ロボット用の簡易衝突判定（NumPyだけで完結）
# 衝突判定 / 干渉判定（collision detection：物体どうしがぶつかっていないかを調べる処理）

# ライブラリの読み込み
import numpy as np

# 自作モジュールの読み込み
from constant import *      # 定数
from grid_map import GridMap   # 占有格子地図


def is_collision_point(grid_map: GridMap, point: np.ndarray) -> bool:
    """
    連続座標の点が障害物に入っていないか（点ロボットの衝突判定）

    パラメータ
        grid_map: 占有格子地図
        point: 連続座標 (x, y)

    戻り値
        is_collision: True / False = 衝突あり（障害物・範囲外） / 衝突なし
    """
    cell = grid_map.pos_to_cell(point)
    is_collision = grid_map.is_obstacle(cell)
    return is_collision


def is_collision_segment(grid_map: GridMap, point1: np.ndarray, point2: np.ndarray,
                         step: float = 0.2) -> bool:
    """
    2点を結ぶ線分が障害物をまたがないか（2点のあいだを細かく刻んで判定する）

    パラメータ
        grid_map: 占有格子地図
        point1: 線分の端点1 (x, y)
        point2: 線分の端点2 (x, y)
        step: 刻み幅（小さいほど厳密だが遅い）

    戻り値
        is_collision: True / False = 衝突あり / 衝突なし
    """
    point1 = np.asarray(point1, dtype=float)
    point2 = np.asarray(point2, dtype=float)

    # 2点間の差分と距離を計算
    difference = point2 - point1
    distance   = np.linalg.norm(difference)

    # 2点間の分割数を算出（最低でも1分割）
    n_divided = max(int(distance / step), 1)
    for i in range(n_divided + 1):
        # 2点間を少しずつ進みながら衝突判定
        divided_point = point1 + i / n_divided * difference
        if is_collision_point(grid_map, divided_point):
            # 1か所でもぶつかれば，この線分は衝突あり
            return True

    return False
