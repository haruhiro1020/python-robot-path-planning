# 経路生成手法である A*（エースター）法の実装
# A*（ダイクストラ法の優先度キューのキーを g（実コスト）から f = g + h（実コスト＋推定コスト）に変え，
#     ゴール方向へ探索を誘導して速く解くアルゴリズム）

# ライブラリの読み込み
import heapq    # 優先度キュー（取り出すたびに最小コストの要素が出るデータ構造）

# 自作モジュールの読み込み
from constant import *          # 定数
from grid_map import GridMap    # 占有格子地図


def heuristic(cell: tuple, goal: tuple, kind: HEURISTIC = HEURISTIC.EUCLIDEAN) -> float:
    """
    ヒューリスティック h(n)：セルからゴールまでのおおよその距離を見積もる

    パラメータ
        cell: 対象のセル (row, col)
        goal: ゴールセル (row, col)
        kind: ヒューリスティックの種類（マンハッタン / ユークリッド）

    戻り値
        h: ゴールまでの推定コスト（0以上）
    """
    d_row = abs(cell[0] - goal[0])
    d_col = abs(cell[1] - goal[1])
    if kind == HEURISTIC.MANHATTAN:
        # マンハッタン距離（縦横の移動量の和）
        return d_row + d_col
    # ユークリッド距離（まっすぐ測った直線距離）
    return (d_row ** 2 + d_col ** 2) ** 0.5


def _reconstruct_path(came_from: dict, start: tuple, goal: tuple) -> list:
    """
    親ノードの記録をたどって，スタートからゴールまでの経路を復元する

    パラメータ
        came_from: 各ノードの親ノードを記録した辞書
        start: スタートセル (row, col)
        goal: ゴールセル (row, col)

    戻り値
        path: スタートからゴールまでのセルのリスト（到達できなければ空リスト）
    """
    if goal not in came_from:
        # ゴールに到達できなかった
        return []

    # ゴールから親をたどり，最後に反転して「始点 → 終点」の順にする
    path = [goal]
    current = goal
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(grid_map: GridMap, connectivity: CONNECTIVITY = CONNECTIVITY.EIGHT,
          heuristic_type: HEURISTIC = HEURISTIC.EUCLIDEAN, weight: float = 1.0) -> tuple:
    """
    A* 法でスタートからゴールまでの最短経路を求める

    パラメータ
        grid_map: 占有格子地図（スタート・ゴール設定済み）
        connectivity: 近傍の種類（4近傍 / 8近傍）
        heuristic_type: ヒューリスティックの種類（マンハッタン / ユークリッド）
        weight: ヒューリスティックの重み（1.0 で通常の A*。大きくすると速いが最適性が崩れる weighted A*）

    戻り値
        path: 最短経路（セルのリスト。到達できなければ空リスト）
        explored: 確定した順に並べた探索済みセルのリスト（可視化・比較用）
        cost: ゴールまでの最短コスト（到達できなければ無限大）
    """
    start = grid_map.start
    goal  = grid_map.goal

    # g 値（始点からの実コスト最小値）と，経路復元用の親ノード
    dist      = {start: 0.0}
    came_from = {}

    # 確定済みノード（コストが二度と更新されないノード）の集合と，確定順
    visited  = set()
    explored = []

    # 優先度キュー（f = g + h の小さいノードから取り出す）。要素は (f, g, セル)
    # ★ダイクストラ法との違いはここだけ：キューのキーが g から f = g + h に変わる
    start_f = heuristic(start, goal, heuristic_type) * weight
    queue = [(start_f, 0.0, start)]

    while queue:
        # いま未確定の中で f が最小のノードを取り出す
        f, g, current = heapq.heappop(queue)
        if current in visited:
            # すでに確定済み（より小さいコストで処理済み）なので飛ばす
            continue
        visited.add(current)
        explored.append(current)

        if current == goal:
            # ゴールを確定したら終了
            break

        # 近傍ノードを緩和（relax）する
        for n_cell, move_cost in grid_map.neighbors(current, connectivity):
            new_g = g + move_cost
            # g(u) + w(u,v) < g(v) なら，より短い経路が見つかったので更新
            if n_cell not in dist or new_g < dist[n_cell]:
                dist[n_cell]      = new_g
                came_from[n_cell] = current
                # f = g + h（h はゴールへの推定。重み weight 倍して誘導の強さを調整）
                new_f = new_g + heuristic(n_cell, goal, heuristic_type) * weight
                heapq.heappush(queue, (new_f, new_g, n_cell))

    # 経路を復元
    path = _reconstruct_path(came_from, start, goal)
    cost = dist.get(goal, float("inf"))

    return path, explored, cost
