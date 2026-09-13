# 経路生成手法である RRT（Rapidly-exploring Random Trees：ランダムに木を広げて経路を探すアルゴリズム）の実装
# RRT（スタートから木〔ツリー〕を1本ずつ伸ばし，自由空間をランダムに埋めるように
#      広げていく。ゴール付近までノードが届いたら，親をたどって経路を復元する手法）

# ライブラリの読み込み
import numpy as np    # 数値計算

# 自作モジュールの読み込み
from constant import *                              # 定数
from grid_map import GridMap                        # 占有格子地図
from collision import is_collision_point, is_collision_segment   # 衝突判定


class Tree:
    """
    探索木（スタートから伸ばしていくノードと，その親子関係を持つツリー）

    PRM の Roadmap は「点どうしを自由につなぐネットワーク」だったが，
    Tree は「各ノードが親を1つだけ持つ木構造」（経路復元のため親をたどれる）。

    プロパティ
        _nodes(list): ノードの連続座標 (x, y) を追加順に並べたリスト
        _parents(list): 各ノードの親ノード番号（根＝スタートだけ親なし -1）

    メソッド
        public
            nodes(): _nodesプロパティのゲッター（numpy配列で返す）
            n_nodes(): ノード数を取得
            add_node(): 親を指定してノードを1つ追加し，その番号を返す
            nearest(): 与えた点に最も近いノードの番号を取得
            edge_list(): 描画用に (親座標, 子座標) の枝リストを取得
            path_to(): 指定ノードからスタートまで親をたどった経路を取得
    """

    def __init__(self, root: np.ndarray) -> None:
        """
        コンストラクタ

        パラメータ
            root: 木の根（スタート）の連続座標 (x, y)
        """
        # 根（スタート）を0番ノードとして登録する。根は親を持たないので -1
        self._nodes   = [np.asarray(root, dtype=float)]
        self._parents = [-1]

    @property
    def nodes(self) -> np.ndarray:
        """
        _nodesプロパティのゲッター（描画・距離計算しやすいよう numpy 配列で返す）
        """
        return np.array(self._nodes)

    @property
    def n_nodes(self) -> int:
        """
        ノード数を取得
        """
        return len(self._nodes)

    def add_node(self, point: np.ndarray, parent_index: int) -> int:
        """
        親を指定してノードを1つ追加し，その番号を返す

        パラメータ
            point: 追加するノードの連続座標 (x, y)
            parent_index: 親ノードの番号（この親から枝が伸びる）

        戻り値
            new_index: 追加したノードの番号
        """
        self._nodes.append(np.asarray(point, dtype=float))
        self._parents.append(parent_index)
        new_index = len(self._nodes) - 1
        return new_index

    def nearest(self, point: np.ndarray) -> int:
        """
        与えた点に最も近いノードの番号を取得（最近傍探索）

        パラメータ
            point: 連続座標 (x, y)

        戻り値
            nearest_index: もっとも近いノードの番号
        """
        # 全ノードとの距離を一括計算し，最小のノードを選ぶ
        nodes     = np.array(self._nodes)
        distances = np.linalg.norm(nodes - point, axis=1)
        nearest_index = int(np.argmin(distances))
        return nearest_index

    def edge_list(self) -> list:
        """
        描画用に (親座標, 子座標) の枝リストを取得

        戻り値
            segments: [((x_parent, y_parent), (x_child, y_child)), ...] のリスト
        """
        segments = []
        for index, parent_index in enumerate(self._parents):
            if parent_index < 0:
                # 根（スタート）は親を持たないので枝はない
                continue
            segments.append((self._nodes[parent_index], self._nodes[index]))
        return segments

    def path_to(self, index: int) -> list:
        """
        指定ノードからスタート（根）まで親をたどり，スタート→指定ノードの順に並べた経路を取得

        パラメータ
            index: 経路の終端にするノードの番号

        戻り値
            path: 経路（連続座標 (x, y) のリスト）
        """
        path = []
        current = index
        while current != -1:
            # 親をたどると終端→…→根の順になるので，最後に反転する
            path.append(self._nodes[current])
            current = self._parents[current]
        path.reverse()
        return path


def sample(grid_map: GridMap, goal_pos: np.ndarray, goal_sample_rate: float,
           rng: np.random.Generator) -> np.ndarray:
    """
    次に木を伸ばす目標点をランダムに選ぶ（ゴールバイアスつき）

    確率 goal_sample_rate でゴールそのものを返し，木をゴール方向へ引き寄せる。
    それ以外は自由空間から一様にサンプルする。

    パラメータ
        grid_map: 占有格子地図
        goal_pos: ゴールの連続座標 (x, y)
        goal_sample_rate: ゴールをサンプルする確率（ゴールバイアス）
        rng: 乱数生成器（シードを固定して再現性を持たせる）

    戻り値
        point: サンプルした連続座標 (x, y)
    """
    if rng.random() < goal_sample_rate:
        # ゴールバイアス：一定確率でゴールそのものを狙う
        return np.asarray(goal_pos, dtype=float)

    # それ以外は地図全体から一様に1点を引く（x は列方向，y は行方向）
    rows, cols = grid_map.shape
    x = rng.uniform(0.0, cols - 1)
    y = rng.uniform(0.0, rows - 1)
    return np.array([x, y])


def steer(from_point: np.ndarray, to_point: np.ndarray, step_size: float) -> np.ndarray:
    """
    from_point から to_point の向きへ，step_size だけ枝を伸ばした新しい点を求める

    to_point が近ければそこまで，遠ければステップ幅で頭打ちにする
    （木が一気に飛ばず，少しずつ伸びるようにするための工夫）。

    パラメータ
        from_point: 枝の根もと（最近傍ノード）の連続座標 (x, y)
        to_point: 伸ばしたい目標点の連続座標 (x, y)
        step_size: 1回で伸ばす距離（ステップ幅）

    戻り値
        new_point: 新しく伸ばした先の連続座標 (x, y)
    """
    difference = np.asarray(to_point, dtype=float) - np.asarray(from_point, dtype=float)
    distance   = np.linalg.norm(difference)
    if distance <= step_size:
        # 目標点までステップ幅より近ければ，そのまま目標点まで伸ばす
        return np.asarray(to_point, dtype=float)
    # 遠ければ目標方向へステップ幅ぶんだけ進む
    new_point = from_point + difference / distance * step_size
    return new_point


def plan_rrt(grid_map: GridMap, max_iterations: int = RRT_MAX_ITERATIONS,
             step_size: float = RRT_STEP_SIZE,
             goal_sample_rate: float = RRT_GOAL_SAMPLE_RATE,
             goal_tolerance: float = RRT_GOAL_TOLERANCE,
             segment_step: float = RRT_SEGMENT_STEP,
             seed: int = RRT_RANDOM_SEED) -> tuple:
    """
    RRT でスタートからゴールまでの経路を生成する

    4ステップ（サンプリング → 最近傍 → 伸ばす → 衝突チェック）を繰り返し，
    最新ノードがゴール付近に届いたら親をたどって経路を復元する。

    パラメータ
        grid_map: 占有格子地図（スタート・ゴール設定済み）
        max_iterations: 木を伸ばす試行の最大回数
        step_size: 1回で伸ばす距離（ステップ幅）
        goal_sample_rate: ゴールをサンプルする確率（ゴールバイアス）
        goal_tolerance: ゴール到達とみなす距離
        segment_step: 枝（線分）の衝突判定の刻み幅
        seed: 乱数シード（同じ結果を再現するために固定する）

    戻り値
        path: 経路（連続座標 (x, y) のリスト。到達できなければ空リスト）
        tree: 成長させた探索木
        n_iterations: ゴール到達までに要した試行回数（失敗時は max_iterations）
    """
    rng = np.random.default_rng(seed)

    start_pos = grid_map.cell_to_pos(grid_map.start)
    goal_pos  = grid_map.cell_to_pos(grid_map.goal)

    # スタートを根として木を初期化
    tree = Tree(start_pos)

    for i in range(1, max_iterations + 1):
        # ステップ1：ランダムサンプリング（ゴールバイアスつき）
        rand_point = sample(grid_map, goal_pos, goal_sample_rate, rng)

        # ステップ2：木の中で最も近いノードを探す
        nearest_index = tree.nearest(rand_point)
        nearest_point = tree.nodes[nearest_index]

        # ステップ3：最近傍からサンプル点の向きへ，ステップ幅だけ枝を伸ばす
        new_point = steer(nearest_point, rand_point, step_size)

        # ステップ4：枝（最近傍→新ノード）が障害物をまたがないか衝突チェック
        if is_collision_segment(grid_map, nearest_point, new_point, segment_step):
            # ぶつかる枝は木に加えない（次の試行へ）
            continue

        # 衝突しなければ木に追加
        new_index = tree.add_node(new_point, nearest_index)

        # ゴール判定：新ノードがゴール付近に届き，かつゴールまで直進できれば到達
        if np.linalg.norm(new_point - goal_pos) <= goal_tolerance:
            if not is_collision_segment(grid_map, new_point, goal_pos, segment_step):
                # ゴールを最後のノードとして加え，経路を復元する
                goal_index = tree.add_node(goal_pos, new_index)
                path = tree.path_to(goal_index)
                return path, tree, i

    # 最大試行回数まで届かなかった（経路なし）
    return [], tree, max_iterations


if __name__ == "__main__":
    # 動作確認（共通サンプル地図で木を伸ばし，経路が引けるか確かめる）
    from grid_map import make_sample_map

    grid_map = make_sample_map()
    path, tree, n_iterations = plan_rrt(grid_map)
    print(f"試行回数 = {n_iterations} / 木のノード数 = {tree.n_nodes}")
    if path:
        print(f"到達 = True / 経路の通過点数 = {len(path)}")
    else:
        print("到達 = False（最大試行回数まで木がゴールへ届かなかった）")
