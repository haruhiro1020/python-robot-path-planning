# 経路生成手法であるダイクストラ法の実装
# ダイクストラ法（始点からの実コスト g が最小のノードから順に確定させ，最短経路を保証する探索アルゴリズム）

# ライブラリの読み込み
import heapq    # 優先度キュー（取り出すたびに最小コストの要素が出るデータ構造）

# 自作モジュールの読み込み
from constant import *          # 定数
from grid_map import GridMap    # 占有格子地図


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


def dijkstra(grid_map: GridMap, connectivity: CONNECTIVITY = CONNECTIVITY.EIGHT) -> tuple:
    """
    ダイクストラ法でスタートからゴールまでの最短経路を求める

    パラメータ
        grid_map: 占有格子地図（スタート・ゴール設定済み）
        connectivity: 近傍の種類（4近傍 / 8近傍）

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

    # 優先度キュー（コストの小さいノードから取り出す）。要素は (コスト, セル)
    queue = [(0.0, start)]

    while queue:
        # いま未確定の中で最小コストのノードを取り出す
        cost, current = heapq.heappop(queue)
        if current in visited:
            # すでに確定済み（より小さいコストで処理済み）なので飛ばす
            continue
        visited.add(current)
        explored.append(current)

        if current == goal:
            # ゴールを確定したら終了（これ以上小さいコストでは来られない）
            break

        # 近傍ノードを緩和（relax）する
        for n_cell, move_cost in grid_map.neighbors(current, connectivity):
            new_cost = cost + move_cost
            # g(u) + w(u,v) < g(v) なら，より短い経路が見つかったので更新
            if n_cell not in dist or new_cost < dist[n_cell]:
                dist[n_cell]      = new_cost
                came_from[n_cell] = current
                heapq.heappush(queue, (new_cost, n_cell))

    # 経路を復元
    path = _reconstruct_path(came_from, start, goal)
    cost = dist.get(goal, float("inf"))

    return path, explored, cost
