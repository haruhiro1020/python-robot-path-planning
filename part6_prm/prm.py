# 経路生成手法である PRM（Probabilistic Roadmap：確率的ロードマップ法）の実装
# PRM（自由空間にランダムな点を撒き，近い点どうしを衝突しない辺でつないで
#      「ロードマップ」〔通行可能な点と道のネットワーク〕を作る。一度作れば
#      同じ地図で何度でも〔多クエリ〕最短経路を引ける手法）

# ライブラリの読み込み
import heapq          # 優先度キュー（取り出すたびに最小コストの要素が出るデータ構造）
import numpy as np    # 数値計算

# 自作モジュールの読み込み
from constant import *                                   # 定数
from grid_map import GridMap                             # 占有格子地図
from collision import is_collision_point, is_collision_segment   # 衝突判定


class Roadmap:
    """
    ロードマップ（ランダムに撒いた点＝ノードと，衝突しない辺＝エッジのネットワーク）

    プロパティ
        _nodes(numpy.ndarray): ノードの連続座標 (x, y) を並べた配列（shape=(ノード数, 2)）
        _edges(dict): 隣接リスト {ノード番号: [(隣のノード番号, 辺の長さ), ...]}

    メソッド
        public
            nodes(): _nodesプロパティのゲッター
            edges(): _edgesプロパティのゲッター
            n_nodes(): ノード数を取得
            n_edges(): 辺の本数（無向辺として数える）を取得
            add_edge(): ノード間に双方向の辺を1本追加
            edge_list(): 描画用に (始点座標, 終点座標) の辺リストを取得
    """

    def __init__(self, nodes: np.ndarray) -> None:
        """
        コンストラクタ

        パラメータ
            nodes: ノードの連続座標 (x, y) を並べた配列（shape=(ノード数, 2)）
        """
        self._nodes = np.asarray(nodes, dtype=float)
        # 隣接リストを空で初期化（各ノードに，つながった相手をあとから追加していく）
        self._edges = {i: [] for i in range(len(self._nodes))}

    @property
    def nodes(self) -> np.ndarray:
        """
        _nodesプロパティのゲッター
        """
        return self._nodes

    @property
    def edges(self) -> dict:
        """
        _edgesプロパティのゲッター
        """
        return self._edges

    @property
    def n_nodes(self) -> int:
        """
        ノード数を取得
        """
        return len(self._nodes)

    @property
    def n_edges(self) -> int:
        """
        辺の本数（無向辺として数える）を取得
        """
        # 隣接リストでは1本の辺を両方向に2回持つので，2で割って実際の本数にする
        return sum(len(neighbors) for neighbors in self._edges.values()) // 2

    def add_edge(self, i: int, j: int, cost: float) -> None:
        """
        ノード i とノード j のあいだに双方向の辺を1本追加

        パラメータ
            i: ノード番号
            j: ノード番号
            cost: 辺の長さ（i と j のユークリッド距離）
        """
        self._edges[i].append((j, cost))
        self._edges[j].append((i, cost))

    def edge_list(self) -> list:
        """
        描画用に (始点座標, 終点座標) の辺リストを取得（重複は除く）

        戻り値
            segments: [((x1, y1), (x2, y2)), ...] のリスト
        """
        segments = []
        for i, neighbors in self._edges.items():
            for j, _cost in neighbors:
                if i < j:
                    # i < j のときだけ追加して，同じ辺を二重に描かないようにする
                    segments.append((self._nodes[i], self._nodes[j]))
        return segments


def sample_free_points(grid_map: GridMap, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    自由空間（障害物のない通行できる場所）にランダムな点を撒く

    障害物に当たった点は捨てて撒き直す（リジェクションサンプリング：条件を
    満たさないサンプルを棄却して，満たすものだけ採用する方法）。

    パラメータ
        grid_map: 占有格子地図
        n_samples: 撒きたい点の数 N
        rng: 乱数生成器（シードを固定して再現性を持たせる）

    戻り値
        points: 自由空間に撒いた点の連続座標 (x, y) を並べた配列（shape=(n_samples, 2)）
    """
    rows, cols = grid_map.shape
    points = []
    while len(points) < n_samples:
        # x は列方向 [0, cols-1]，y は行方向 [0, rows-1] の範囲で一様乱数を引く
        x = rng.uniform(0.0, cols - 1)
        y = rng.uniform(0.0, rows - 1)
        point = np.array([x, y])
        if not is_collision_point(grid_map, point):
            # 障害物に入っていない点だけ採用する
            points.append(point)
    return np.array(points)


def build_roadmap(grid_map: GridMap, points: np.ndarray,
                  connection_radius: float = PRM_CONNECTION_RADIUS,
                  segment_step: float = PRM_SEGMENT_STEP) -> Roadmap:
    """
    撒いた点から，近い点どうしを「衝突しない辺」でつないでロードマップを構築する（構築フェーズ）

    パラメータ
        grid_map: 占有格子地図
        points: 自由空間に撒いた点 (x, y) の配列
        connection_radius: 近傍接続の半径 r（この距離以内の点どうしをつなごうとする）
        segment_step: 辺（線分）の衝突判定の刻み幅

    戻り値
        roadmap: 構築したロードマップ
    """
    roadmap = Roadmap(points)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            # 2点間の距離が接続半径より遠ければつながない
            difference = points[j] - points[i]
            distance   = np.linalg.norm(difference)
            if distance > connection_radius:
                continue
            # 半径内でも，途中に障害物があれば（線分が壁をまたげば）辺は張らない
            if is_collision_segment(grid_map, points[i], points[j], segment_step):
                continue
            roadmap.add_edge(i, j, distance)
    return roadmap


def _astar_on_roadmap(roadmap: Roadmap, start_index: int, goal_index: int) -> tuple:
    """
    ロードマップ（グラフ）の上で A* を走らせ，2ノード間の最短経路を求める（クエリフェーズの中身）

    第4章の A* は「格子の近傍」を探索したが，ここでは探索対象が
    「ロードマップの隣接ノード」に変わるだけで，f = g + h で誘導する考え方は同じ。

    パラメータ
        roadmap: ロードマップ
        start_index: スタートノードの番号
        goal_index: ゴールノードの番号

    戻り値
        index_path: ノード番号で表した最短経路（到達できなければ空リスト）
        cost: ゴールまでの最短コスト（到達できなければ無限大）
    """
    nodes     = roadmap.nodes
    goal_pos  = nodes[goal_index]

    # g 値（始点からの実コスト最小値）と，経路復元用の親ノード
    dist      = {start_index: 0.0}
    came_from = {}
    visited   = set()

    def heuristic(index: int) -> float:
        # h(n)：ノードからゴールまでのまっすぐな距離（ユークリッド距離）で見積もる
        return float(np.linalg.norm(nodes[index] - goal_pos))

    # 優先度キュー（f = g + h の小さいノードから取り出す）。要素は (f, g, ノード番号)
    queue = [(heuristic(start_index), 0.0, start_index)]

    while queue:
        f, g, current = heapq.heappop(queue)
        if current in visited:
            # すでに確定済みなら飛ばす
            continue
        visited.add(current)

        if current == goal_index:
            # ゴールを確定したら終了
            break

        # 隣接ノードを緩和（relax）する
        for neighbor, edge_cost in roadmap.edges[current]:
            new_g = g + edge_cost
            if neighbor not in dist or new_g < dist[neighbor]:
                dist[neighbor]      = new_g
                came_from[neighbor] = current
                new_f = new_g + heuristic(neighbor)
                heapq.heappush(queue, (new_f, new_g, neighbor))

    # 経路を復元（ゴールから親をたどり，反転して始点→終点の順にする）
    if goal_index not in dist:
        return [], float("inf")
    index_path = [goal_index]
    current = goal_index
    while current != start_index:
        current = came_from[current]
        index_path.append(current)
    index_path.reverse()
    return index_path, dist[goal_index]


def _connect_terminal(grid_map: GridMap, roadmap: Roadmap, point: np.ndarray,
                      connection_radius: float, segment_step: float) -> int:
    """
    スタート／ゴールの点をロードマップに新しいノードとして追加し，近傍へ接続する

    パラメータ
        grid_map: 占有格子地図
        roadmap: 接続先のロードマップ（ノードが追加される）
        point: 追加する点の連続座標 (x, y)
        connection_radius: 近傍接続の半径 r
        segment_step: 辺の衝突判定の刻み幅

    戻り値
        new_index: 追加したノードの番号
    """
    # ノード配列の末尾に point を足し，隣接リストにも空の枠を用意する
    new_index = roadmap.n_nodes
    roadmap._nodes = np.vstack([roadmap.nodes, point])
    roadmap._edges[new_index] = []

    # 既存ノードのうち，半径内かつ衝突しないものへ辺を張る
    for i in range(new_index):
        difference = roadmap.nodes[i] - point
        distance   = np.linalg.norm(difference)
        if distance > connection_radius:
            continue
        if is_collision_segment(grid_map, point, roadmap.nodes[i], segment_step):
            continue
        roadmap.add_edge(new_index, i, distance)
    return new_index


def plan_prm(grid_map: GridMap, n_samples: int = PRM_N_SAMPLES,
             connection_radius: float = PRM_CONNECTION_RADIUS,
             segment_step: float = PRM_SEGMENT_STEP,
             seed: int = PRM_RANDOM_SEED) -> tuple:
    """
    PRM でスタートからゴールまでの経路を生成する（構築フェーズ＋クエリフェーズ）

    パラメータ
        grid_map: 占有格子地図（スタート・ゴール設定済み）
        n_samples: 自由空間に撒くサンプル点の数 N
        connection_radius: 近傍接続の半径 r
        segment_step: 辺の衝突判定の刻み幅
        seed: 乱数シード（同じ結果を再現するために固定する）

    戻り値
        path: 経路（連続座標 (x, y) のリスト。到達できなければ空リスト）
        roadmap: 構築したロードマップ（スタート・ゴールも接続済み）
        cost: ゴールまでの経路コスト（到達できなければ無限大）
    """
    rng = np.random.default_rng(seed)

    # --- 構築フェーズ：点を撒いて，近傍を衝突しない辺でつなぐ ---
    points  = sample_free_points(grid_map, n_samples, rng)
    roadmap = build_roadmap(grid_map, points, connection_radius, segment_step)

    # --- クエリフェーズ：スタート・ゴールを接続し，A* で最短を引く ---
    start_pos = grid_map.cell_to_pos(grid_map.start)
    goal_pos  = grid_map.cell_to_pos(grid_map.goal)
    start_index = _connect_terminal(grid_map, roadmap, start_pos, connection_radius, segment_step)
    goal_index  = _connect_terminal(grid_map, roadmap, goal_pos,  connection_radius, segment_step)

    index_path, cost = _astar_on_roadmap(roadmap, start_index, goal_index)

    # ノード番号の経路を，連続座標 (x, y) の経路へ変換
    path = [roadmap.nodes[index] for index in index_path]
    return path, roadmap, cost


if __name__ == "__main__":
    # 動作確認（共通サンプル地図でロードマップを作り，経路が引けるか確かめる）
    from grid_map import make_sample_map

    grid_map = make_sample_map()
    path, roadmap, cost = plan_prm(grid_map)
    print(f"ノード数 = {roadmap.n_nodes} / 辺の本数 = {roadmap.n_edges}")
    if path:
        print(f"到達 = True / 経由ノード数 = {len(path)} / 経路コスト = {cost:.2f}")
    else:
        print("到達 = False（ロードマップ上でスタートとゴールがつながらなかった）")
