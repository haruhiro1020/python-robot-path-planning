# BFS / ダイクストラ法 / Greedy Best-First / A* の4手法を「g と h の役割」で見比べるための実装
# 4手法は次の関係で整理できる（g＝始点からの実コスト，h＝ゴールへの推定コスト）
#   BFS      … コストを無視して近い順（手数の少ない順）に均等展開
#   ダイクストラ法 … g だけで並べる（最短保証はあるがゴール方向に誘導されない）
#   Greedy   … h だけで並べる（ゴールへ一直線だが最短保証なし）
#   A*       … f = g + h で並べる（誘導されつつ最短も保証）

# ライブラリの読み込み
import heapq                 # 優先度キュー
from collections import deque    # FIFO キュー（先に入れたものから取り出す。BFS 用）

# 自作モジュールの読み込み
from constant import *          # 定数
from grid_map import GridMap    # 占有格子地図
from astar import heuristic, _reconstruct_path    # ヒューリスティックと経路復元（A* と共有）


def bfs(grid_map: GridMap, connectivity: CONNECTIVITY = CONNECTIVITY.EIGHT) -> tuple:
    """
    幅優先探索（BFS：Breadth-First Search。コストを無視し，手数の少ない順に均等に広げる）

    パラメータ
        grid_map: 占有格子地図（スタート・ゴール設定済み）
        connectivity: 近傍の種類（4近傍 / 8近傍）

    戻り値
        path: 経路（セルのリスト。到達できなければ空リスト）
        explored: 確定した順に並べた探索済みセルのリスト
        cost: 経路の手数（セル数 - 1。到達できなければ無限大）
    """
    start = grid_map.start
    goal  = grid_map.goal

    came_from = {}
    visited   = {start}
    explored  = []

    # FIFO キュー（先に入れたものから取り出す → 近いセルから順に広がる）
    queue = deque([start])

    while queue:
        current = queue.popleft()
        explored.append(current)
        if current == goal:
            break
        for n_cell, _ in grid_map.neighbors(current, connectivity):
            if n_cell not in visited:
                # 初めて到達したセルだけを記録（BFS は手数最小で最初に到達する）
                visited.add(n_cell)
                came_from[n_cell] = current
                queue.append(n_cell)

    path = _reconstruct_path(came_from, start, goal)
    cost = (len(path) - 1) if path else float("inf")
    return path, explored, cost


def priority_search(grid_map: GridMap, g_weight: float, h_weight: float,
                    connectivity: CONNECTIVITY = CONNECTIVITY.EIGHT,
                    heuristic_type: HEURISTIC = HEURISTIC.EUCLIDEAN) -> tuple:
    """
    優先度キューを使う探索の共通実装（重みでダイクストラ法 / Greedy / A* を切り替える）
    キー = g_weight * g + h_weight * h
        ダイクストラ法 : g_weight=1, h_weight=0
        Greedy   : g_weight=0, h_weight=1
        A*       : g_weight=1, h_weight=1

    パラメータ
        grid_map: 占有格子地図（スタート・ゴール設定済み）
        g_weight: g（始点からの実コスト）の重み
        h_weight: h（ゴールへの推定コスト）の重み
        connectivity: 近傍の種類（4近傍 / 8近傍）
        heuristic_type: ヒューリスティックの種類

    戻り値
        path: 経路（セルのリスト。到達できなければ空リスト）
        explored: 確定した順に並べた探索済みセルのリスト
        cost: ゴールまでの実コスト g（到達できなければ無限大）
    """
    start = grid_map.start
    goal  = grid_map.goal

    dist      = {start: 0.0}
    came_from = {}
    visited   = set()
    explored  = []

    start_key = h_weight * heuristic(start, goal, heuristic_type)
    queue = [(start_key, 0.0, start)]

    while queue:
        _, g, current = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        explored.append(current)
        if current == goal:
            break
        for n_cell, move_cost in grid_map.neighbors(current, connectivity):
            new_g = g + move_cost
            if n_cell not in dist or new_g < dist[n_cell]:
                dist[n_cell]      = new_g
                came_from[n_cell] = current
                key = g_weight * new_g + h_weight * heuristic(n_cell, goal, heuristic_type)
                heapq.heappush(queue, (key, new_g, n_cell))

    path = _reconstruct_path(came_from, start, goal)
    cost = dist.get(goal, float("inf"))
    return path, explored, cost
