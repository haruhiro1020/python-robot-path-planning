# 2軸アーム用の衝突判定（連続的な関節空間で python-fcl を使う）
#
# 第2章の collision.py は「点が障害物セルに入ったか」を NumPy だけで調べていました。
# 本章では同じ関数名・同じ呼び出し方のまま，中身を「その関節角度のときアームが障害物に
# ぶつかるか」を python-fcl（Flexible Collision Library：図形どうしの衝突を高速に判定する
# ライブラリ）で調べるものに差し替えます。関数のインターフェースをそろえているので，
# 第6〜10章のサンプリング系アルゴリズム（PRM/RRT/RRT-Connect/RRT*/Informed RRT*）の
# コードは，1行も変えずにそのまま関節空間で動きます。
#
# サンプリング系は格子に丸めず「連続的な関節空間」を直接探索する。座標 (x, y) は
# CSpaceMap.coord_to_joint で関節角度 (θ1, θ2) に直してから fcl で判定する。

# ライブラリの読み込み
import numpy as np

# 自作モジュールの読み込み
from constant import *              # 定数
from grid_map import CSpaceMap      # 関節空間（C-space）地図


def is_collision_point(cspace_map: CSpaceMap, point: np.ndarray) -> bool:
    """
    連続座標の点（＝ある関節角度）で，アームが障害物にぶつからないか

    点ロボット版（第2章）と同じ呼び出し方。中身だけ「座標→関節角度→fcl干渉判定」に変わる。

    パラメータ
        cspace_map: 関節空間（C-space）地図（アーム・環境を保持）
        point: 連続座標 (x, y)（関節空間上の1点）

    戻り値
        is_collision: True / False = ぶつかる（禁止領域） / ぶつからない
    """
    # 座標 (x, y) を関節角度 (θ1, θ2) に直す
    joint = cspace_map.coord_to_joint(point)
    # アームをその角度に動かし，環境と干渉判定する
    cspace_map.robot.update(joint)
    return cspace_map.environment.is_collision_dist(cspace_map.robot.manager,
                                                    margin=COLLISION_MARGIN)


def is_collision_segment(cspace_map: CSpaceMap, point1: np.ndarray, point2: np.ndarray,
                         step: float = 0.2) -> bool:
    """
    2点（2つの関節角度）を結ぶ線分の途中で，アームが障害物にぶつからないか

    関節空間で2点を結ぶ直線は「2つの姿勢のあいだを関節角度が一定の割合で変わる動き」に対応する。
    その途中を細かく刻んで，1か所でもぶつかれば衝突ありとする（点ロボット版と同じ考え方）。

    パラメータ
        cspace_map: 関節空間（C-space）地図
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
        if is_collision_point(cspace_map, divided_point):
            # 1か所でもぶつかれば，この線分は衝突あり
            return True

    return False
