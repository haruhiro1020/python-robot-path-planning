# 経路生成手法である RRT*（RRTスター：RRTに「最良の親選び〔choose parent〕」と
#   「つなぎ直し〔rewire〕」を足し，反復するほど経路を最適へ近づけるアルゴリズム）の実装
# RRT*（新ノードを追加するとき，半径内の近傍の中から「始点からのコストが最小になる親」を選び，
#       さらに新ノード経由で安くなる近傍の親を新ノードへ付け替える〔rewire〕。これを繰り返すと
#       経路コストが単調に下がり，サンプルを増やすほど最適経路へ収束する〔漸近最適性〕）

# ライブラリの読み込み
import numpy as np    # 数値計算

# 自作モジュールの読み込み
from constant import *                              # 定数
from grid_map import GridMap                        # 占有格子地図
from collision import is_collision_segment          # 線分の衝突判定
# 第7章 RRT の部品をそのまま再利用する（木構造・一様サンプリング・ステップ伸長）
from rrt import Tree, sample, steer


class StarTree(Tree):
    """
    RRT* 用の探索木（第7章 Tree に「コスト」と「親の付け替え」を足したもの）

    RRT の Tree は親子関係だけを持っていたが，RRT* では各ノードに
    「始点からの累積コスト（経路の長さ）」を持たせ，より安い経路が見つかったら
    親を付け替え（rewire）てコストを下げていく。

    プロパティ（Tree から追加）
        _costs(list): 各ノードの始点からの累積コスト（根＝0.0）

    メソッド
        public
            cost(): 指定ノードの始点からの累積コストを取得
            add_node(): 親を指定してノードを追加（コストも一緒に記録）※オーバーライド
            near(): 指定点から半径内にある近傍ノードの番号一覧を取得
            set_parent(): 指定ノードの親を付け替え，コストを更新する（rewire 用）
    """

    def __init__(self, root: np.ndarray) -> None:
        """
        コンストラクタ

        パラメータ
            root: 木の根（スタート）の連続座標 (x, y)
        """
        super().__init__(root)
        # 根（スタート）の始点からのコストは 0
        self._costs = [0.0]

    def cost(self, index: int) -> float:
        """
        指定ノードの始点からの累積コスト（経路の長さ）を取得

        パラメータ
            index: ノード番号

        戻り値
            cost: 始点からそのノードまでの累積コスト
        """
        return self._costs[index]

    def add_node(self, point: np.ndarray, parent_index: int) -> int:
        """
        親を指定してノードを1つ追加し，その番号を返す（コストも記録する）

        コスト ＝ 親のコスト ＋ 親からこのノードまでの距離，として累積で持つ。

        パラメータ
            point: 追加するノードの連続座標 (x, y)
            parent_index: 親ノードの番号

        戻り値
            new_index: 追加したノードの番号
        """
        new_index = super().add_node(point, parent_index)
        # 親までのコストに，親→新ノードの距離（枝の長さ）を足したものが新ノードのコスト
        edge = float(np.linalg.norm(self._nodes[new_index] - self._nodes[parent_index]))
        self._costs.append(self._costs[parent_index] + edge)
        return new_index

    def near(self, point: np.ndarray, radius: float) -> list:
        """
        指定点から半径内にある近傍ノードの番号一覧を取得（choose parent / rewire の対象集め）

        パラメータ
            point: 中心となる連続座標 (x, y)
            radius: 近傍とみなす半径

        戻り値
            indices: 半径内にあるノード番号のリスト
        """
        nodes     = np.array(self._nodes)
        distances = np.linalg.norm(nodes - point, axis=1)
        indices   = [int(i) for i in np.where(distances <= radius)[0]]
        return indices

    def set_parent(self, index: int, parent_index: int) -> None:
        """
        指定ノードの親を付け替え，コストを更新する（rewire：つなぎ直し）

        親を変えるとそのノードのコストが変わり，さらにそのノードを親に持つ子のコストも
        変わるため，子孫へコストの変化を伝播させる。

        パラメータ
            index: 親を付け替えるノードの番号
            parent_index: 新しい親ノードの番号
        """
        self._parents[index] = parent_index
        edge = float(np.linalg.norm(self._nodes[index] - self._nodes[parent_index]))
        self._costs[index] = self._costs[parent_index] + edge
        # 親が変わったぶん，このノードを根とする部分木のコストもまとめて更新する
        self._propagate_cost(index)

    def _propagate_cost(self, index: int) -> None:
        """
        指定ノードを親に持つ子へ，コストの変化を再帰的に伝える（内部用）

        パラメータ
            index: コストが変わった親ノードの番号
        """
        for child, parent in enumerate(self._parents):
            if parent == index:
                edge = float(np.linalg.norm(self._nodes[child] - self._nodes[index]))
                self._costs[child] = self._costs[index] + edge
                self._propagate_cost(child)


def neighbor_radius(n_nodes: int, gamma: float = RRT_STAR_NEIGHBOR_GAMMA,
                    dimension: int = DIMENSION_2D,
                    max_radius: float = RRT_STAR_NEIGHBOR_MAX) -> float:
    """
    ノード数に応じた近傍半径 r_n = γ・(log n / n)^(1/d) を求める

    ノードが増えるほど半径を縮め，choose parent / rewire の対象を絞る。
    こうすると計算量を抑えつつ漸近最適性（サンプルを増やすほど最適へ収束）を保てる。

    パラメータ
        n_nodes: 現在の木のノード数 n
        gamma: 半径の係数 γ（大きいほど広く探す）
        dimension: 空間の次元 d（点ロボットは2次元）
        max_radius: 半径の上限（序盤に大きくなりすぎるのを防ぐ頭打ち）

    戻り値
        radius: 近傍半径 r_n
    """
    if n_nodes <= 1:
        # ノードが1個（根だけ）のときは log が効かないので上限値を使う
        return max_radius
    radius = gamma * (np.log(n_nodes) / n_nodes) ** (1.0 / dimension)
    # 上限で頭打ちにする（序盤の半径の暴れを抑える）
    return float(min(radius, max_radius))


def choose_parent(grid_map: GridMap, tree: StarTree, new_point: np.ndarray,
                  neighbors: list, default_parent: int, segment_step: float) -> tuple:
    """
    新ノードを，近傍の中で「始点からのコストが最小になる親」につなぐ（choose parent）

    RRT は最近傍ノードを無条件で親にしていたが，RRT* は半径内の近傍を見渡し，
    そこを経由したときのコストが一番小さくなる親を選ぶ（衝突する枝は候補から外す）。

    パラメータ
        grid_map: 占有格子地図
        tree: RRT* 探索木
        new_point: 追加しようとしている新ノードの連続座標 (x, y)
        neighbors: 近傍ノード番号のリスト
        default_parent: 既定の親（最近傍ノード。これは衝突なしと分かっている）
        segment_step: 枝（線分）の衝突判定の刻み幅

    戻り値
        best_parent: 選ばれた親ノードの番号
        best_cost: その親を経由したときの新ノードのコスト
    """
    # まずは最近傍ノードを親にした場合をベースラインにする
    best_parent = default_parent
    best_cost   = tree.cost(default_parent) + \
        float(np.linalg.norm(new_point - tree.nodes[default_parent]))

    for index in neighbors:
        neighbor_point = tree.nodes[index]
        # この近傍を経由したときのコスト（近傍までのコスト＋近傍→新ノードの距離）
        candidate_cost = tree.cost(index) + float(np.linalg.norm(new_point - neighbor_point))
        if candidate_cost >= best_cost:
            # いまの最良より高ければ親候補にしない
            continue
        if is_collision_segment(grid_map, neighbor_point, new_point, segment_step):
            # 枝が障害物をまたぐ近傍は親にできない
            continue
        best_parent, best_cost = index, candidate_cost

    return best_parent, best_cost


def rewire(grid_map: GridMap, tree: StarTree, new_index: int,
           neighbors: list, segment_step: float) -> None:
    """
    新ノード経由で安くなる近傍の親を，新ノードへ付け替える（rewire：つなぎ直し）

    choose parent は「新ノードの親」を最適化したが，rewire は逆に「近傍ノードの親」を見直す。
    新ノードを経由した方がコストが下がる近傍があれば，その親を新ノードへ付け替えてコストを下げる。

    パラメータ
        grid_map: 占有格子地図
        tree: RRT* 探索木
        new_index: 追加したばかりの新ノードの番号
        neighbors: 近傍ノード番号のリスト
        segment_step: 枝（線分）の衝突判定の刻み幅
    """
    new_point = tree.nodes[new_index]
    new_cost  = tree.cost(new_index)

    for index in neighbors:
        if index == new_index:
            continue
        neighbor_point = tree.nodes[index]
        # 新ノードを経由したときの近傍ノードのコスト
        candidate_cost = new_cost + float(np.linalg.norm(neighbor_point - new_point))
        if candidate_cost >= tree.cost(index):
            # いまの親を使う方が安い（または同じ）ならつなぎ直さない
            continue
        if is_collision_segment(grid_map, new_point, neighbor_point, segment_step):
            # 枝が障害物をまたぐならつなぎ直せない
            continue
        # 近傍ノードの親を新ノードへ付け替え，コストを更新する
        tree.set_parent(index, new_index)


def plan_rrt_star(grid_map: GridMap,
                  max_iterations: int = RRT_STAR_MAX_ITERATIONS,
                  step_size: float = RRT_STAR_STEP_SIZE,
                  goal_sample_rate: float = RRT_STAR_GOAL_SAMPLE_RATE,
                  goal_tolerance: float = RRT_STAR_GOAL_TOLERANCE,
                  segment_step: float = RRT_STAR_SEGMENT_STEP,
                  gamma: float = RRT_STAR_NEIGHBOR_GAMMA,
                  max_radius: float = RRT_STAR_NEIGHBOR_MAX,
                  seed: int = RRT_STAR_RANDOM_SEED) -> tuple:
    """
    RRT* でスタートからゴールまでの経路を生成する

    RRT の4ステップ（サンプリング→最近傍→伸ばす→衝突チェック）に，
    choose parent（最良の親選び）と rewire（つなぎ直し）を足す。
    ゴールに届いても探索を止めず，反復のたびに経路コストを下げていく。

    パラメータ
        grid_map: 占有格子地図（スタート・ゴール設定済み）
        max_iterations: 木を伸ばす試行の最大回数（多いほど最適へ近づく）
        step_size: 1回で伸ばす距離（ステップ幅）
        goal_sample_rate: ゴールをサンプルする確率（ゴールバイアス）
        goal_tolerance: ゴール到達とみなす距離
        segment_step: 枝（線分）の衝突判定の刻み幅
        gamma: 近傍半径の係数 γ
        max_radius: 近傍半径の上限
        seed: 乱数シード（同じ結果を再現するために固定する）

    戻り値
        path: 最良の経路（連続座標 (x, y) のリスト。到達できなければ空リスト）
        tree: 成長させた RRT* 探索木
        cost_history: (試行回数, その時点の最良コスト) のリスト（コスト低下のグラフ用）
        snapshots: 最良経路が更新されるたびの (試行回数, コスト, 経路) のリスト（gif用）
    """
    rng = np.random.default_rng(seed)

    start_pos = grid_map.cell_to_pos(grid_map.start)
    goal_pos  = grid_map.cell_to_pos(grid_map.goal)

    # スタートを根として RRT* 木を初期化
    tree = StarTree(start_pos)

    # ゴールへ直進できるノード番号の集まり（ここから最良の経路を選ぶ）
    goal_candidates = []
    cost_history = []   # (試行回数, 最良コスト) … 反復ごとのコスト低下を記録
    snapshots    = []   # (試行回数, コスト, 経路) … 最良経路が更新されたときだけ記録
    best_cost = float("inf")

    for i in range(1, max_iterations + 1):
        # ステップ1：ランダムサンプリング（ゴールバイアスつき）
        rand_point = sample(grid_map, goal_pos, goal_sample_rate, rng)

        # ステップ2：木の中で最も近いノードを探す
        nearest_index = tree.nearest(rand_point)
        nearest_point = tree.nodes[nearest_index]

        # ステップ3：最近傍からサンプル点の向きへ，ステップ幅だけ枝を伸ばす
        new_point = steer(nearest_point, rand_point, step_size)

        # ステップ4：枝（最近傍→新ノード）が障害物をまたぐなら木に加えない
        if is_collision_segment(grid_map, nearest_point, new_point, segment_step):
            continue

        # ステップ5（RRT*）：半径内の近傍を集め，最良の親を選んでから追加する
        radius    = neighbor_radius(tree.n_nodes, gamma, DIMENSION_2D, max_radius)
        neighbors = tree.near(new_point, radius)
        best_parent, _ = choose_parent(grid_map, tree, new_point, neighbors,
                                       nearest_index, segment_step)
        new_index = tree.add_node(new_point, best_parent)

        # ステップ6（RRT*）：新ノード経由で安くなる近傍をつなぎ直す（rewire）
        rewire(grid_map, tree, new_index, neighbors, segment_step)

        # ゴール判定：新ノードがゴール付近に届き，かつゴールまで直進できれば候補に加える
        if np.linalg.norm(new_point - goal_pos) <= goal_tolerance:
            if not is_collision_segment(grid_map, new_point, goal_pos, segment_step):
                goal_candidates.append(new_index)

        # ここまでで見つかっている最良の経路コストを評価する
        #   （rewire で既存候補のコストが下がることもあるので毎回コストを取り直す）
        if goal_candidates:
            costs = [tree.cost(idx) + float(np.linalg.norm(tree.nodes[idx] - goal_pos))
                     for idx in goal_candidates]
            current_best = float(min(costs))
            cost_history.append((i, current_best))
            if current_best < best_cost - EPSILON:
                # 最良コストが更新されたら，その経路を1枚スナップショットとして残す（gif用）
                best_cost = current_best
                best_idx  = goal_candidates[int(np.argmin(costs))]
                path = tree.path_to(best_idx) + [goal_pos]
                snapshots.append((i, current_best, path))

    # 最終的な最良経路を組み立てる（候補がなければ経路なし）
    if goal_candidates:
        costs = [tree.cost(idx) + float(np.linalg.norm(tree.nodes[idx] - goal_pos))
                 for idx in goal_candidates]
        best_idx = goal_candidates[int(np.argmin(costs))]
        path = tree.path_to(best_idx) + [goal_pos]
    else:
        path = []

    return path, tree, cost_history, snapshots


if __name__ == "__main__":
    # 動作確認（共通サンプル地図で RRT* を回し，経路コストが下がるか確かめる）
    from grid_map import make_sample_map

    grid_map = make_sample_map()
    path, tree, cost_history, snapshots = plan_rrt_star(grid_map)
    print(f"木のノード数 = {tree.n_nodes} / 最良経路の更新回数 = {len(snapshots)}")
    if path:
        first_cost = snapshots[0][1]
        last_cost  = snapshots[-1][1]
        print(f"到達 = True / 経路の通過点数 = {len(path)}")
        print(f"初期解コスト = {first_cost:.2f} → 最終コスト = {last_cost:.2f}"
              f"（{first_cost - last_cost:.2f} 短縮）")
    else:
        print("到達 = False（最大試行回数までゴールへ届かなかった）")
