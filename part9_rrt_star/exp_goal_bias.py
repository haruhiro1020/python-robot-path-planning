# 第9章b の「ゴールバイアスを途中で切り替えると効くのか」を実測する検証スクリプト
# 本文の message 欄に載せた表（シード20本の平均）は，このスクリプトの出力である。
#   ・バイアスが効くのは「初期解に届くまで」で，高いほど早く届く
#   ・初期解が出たあとに 0 へ落としても，最終コストはシードのばらつきに埋もれて変わらない
# main.py（本番の図を作る）とは別物なので，画像は一切書き出さない。
#
# 実行: このフォルダに移動して `python exp_goal_bias.py`（20本×5設定で数分かかる）

# ライブラリの読み込み
import multiprocessing as mp   # 並列実行（設定×シードを同時に回す）
import numpy as np             # 数値計算

# 自作モジュールの読み込み
from constant import *                              # 定数
from collision import is_collision_segment          # 線分の衝突判定
from grid_map import GridMap, make_sample_map       # 占有格子地図
from rrt import sample, steer                       # 第7章 RRT の部品
from rrt_star import StarTree, neighbor_radius, choose_parent, rewire

# 比較する設定（表示名, 初期解が出るまでのバイアス, 初期解が出たあとのバイアス）
CONFIGS = [
    ("0.00 固定（バイアスなし）", 0.00, 0.00),
    ("0.05 固定（本書）",         0.05, 0.05),
    ("0.10 固定",                 0.10, 0.10),
    ("0.10 → 到達後に 0.00",      0.10, 0.00),
]
SEEDS = list(range(20))    # 1本のシードでは偶然に左右されるので20本の平均で比べる


def plan(grid_map: GridMap, seed: int, rate_before: float, rate_after: float,
         max_iterations: int = RRT_STAR_MAX_ITERATIONS,
         step_size: float = RRT_STAR_STEP_SIZE,
         goal_tolerance: float = RRT_STAR_GOAL_TOLERANCE,
         segment_step: float = RRT_STAR_SEGMENT_STEP,
         gamma: float = RRT_STAR_NEIGHBOR_GAMMA,
         max_radius: float = RRT_STAR_NEIGHBOR_MAX) -> tuple:
    """
    ゴールバイアスを途中で切り替えられるようにした RRT*（plan_rrt_star の計測用の写し）

    plan_rrt_star との違いはステップ1だけで，初期解が出る前後でバイアスを使い分ける。
    図もスナップショットも要らないので，計測に必要な数字だけを返す。

    パラメータ
        grid_map: 占有格子地図（スタート・ゴール設定済み）
        seed: 乱数シード
        rate_before: 初期解が出るまでのゴールバイアス
        rate_after: 初期解が出たあとのゴールバイアス
        max_iterations: 木を伸ばす試行の最大回数
        step_size: 1回で伸ばす距離（ステップ幅）
        goal_tolerance: ゴール到達とみなす距離
        segment_step: 枝（線分）の衝突判定の刻み幅
        gamma: 近傍半径の係数 γ
        max_radius: 近傍半径の上限

    戻り値
        first_iteration: 初期解が出た試行回数（届かなければ None）
        first_cost: 初期解のコスト（届かなければ None）
        best_cost: 最終的な最良コスト（届かなければ inf）
        n_nodes: 最終的な木のノード数
    """
    rng = np.random.default_rng(seed)

    start_pos = grid_map.cell_to_pos(grid_map.start)
    goal_pos  = grid_map.cell_to_pos(grid_map.goal)

    tree = StarTree(start_pos)
    goal_candidates = []
    best_cost = float("inf")
    first_iteration, first_cost = None, None

    for i in range(1, max_iterations + 1):
        # ステップ1：初期解が出ているかどうかでゴールバイアスを切り替える
        goal_sample_rate = rate_before if not goal_candidates else rate_after
        rand_point = sample(grid_map, goal_pos, goal_sample_rate, rng)

        # ステップ2〜4：最近傍を探して伸ばし，枝が障害物をまたぐなら捨てる
        nearest_index = tree.nearest(rand_point)
        nearest_point = tree.nodes[nearest_index]
        new_point     = steer(nearest_point, rand_point, step_size)
        if is_collision_segment(grid_map, nearest_point, new_point, segment_step):
            continue

        # ステップ5〜6：最良の親を選んで追加し，安くなる近傍をつなぎ直す
        radius    = neighbor_radius(tree.n_nodes, gamma, DIMENSION_2D, max_radius)
        neighbors = tree.near(new_point, radius)
        best_parent, _ = choose_parent(grid_map, tree, new_point, neighbors,
                                       nearest_index, segment_step)
        new_index = tree.add_node(new_point, best_parent)
        rewire(grid_map, tree, new_index, neighbors, segment_step)

        # ゴール判定（plan_rrt_star と同じ）
        if np.linalg.norm(new_point - goal_pos) <= goal_tolerance:
            if not is_collision_segment(grid_map, new_point, goal_pos, segment_step):
                goal_candidates.append(new_index)

        if goal_candidates:
            costs = [tree.cost(idx) + float(np.linalg.norm(tree.nodes[idx] - goal_pos))
                     for idx in goal_candidates]
            current_best = float(min(costs))
            if current_best < best_cost - EPSILON:
                best_cost = current_best
                if first_iteration is None:
                    # 初めてゴールへ届いた時点（初期解）を記録する
                    first_iteration, first_cost = i, current_best

    return first_iteration, first_cost, best_cost, tree.n_nodes


def run_one(job: tuple) -> tuple:
    """
    1設定×1シードを実行する（並列実行のワーカー）

    パラメータ
        job: (設定の表示名, 初期解までのバイアス, 初期解後のバイアス, シード)

    戻り値
        result: (設定の表示名, シード, plan() の戻り値)
    """
    name, rate_before, rate_after, seed = job
    return (name, seed, plan(make_sample_map(), seed, rate_before, rate_after))


def main() -> None:
    """
    メイン処理（設定×シードを並列実行し，平均と対の差を表示する）
    """
    jobs = [(name, before, after, seed)
            for (name, before, after) in CONFIGS for seed in SEEDS]
    # macOS の既定は spawn だが，自作モジュールを読み直さずに済む fork で回す
    with mp.get_context("fork").Pool(8) as pool:
        results = pool.map(run_one, jobs)

    table = {(name, seed): row for name, seed, row in results}

    print(f"=== ゴールバイアスの比較（シード{SEEDS[0]}〜{SEEDS[-1]} の {len(SEEDS)} 本, "
          f"max_iterations={RRT_STAR_MAX_ITERATIONS}）===")
    print(f"{'ゴールバイアス':<26}{'初期解が出た反復':>10}{'初期解コスト':>10}{'最終コスト':>10}")
    for name, _, _ in CONFIGS:
        rows = [table[(name, seed)] for seed in SEEDS]
        reached = [r for r in rows if r[0] is not None]
        if not reached:
            print(f"{name:<26}{'届かず':>10}")
            continue
        print(f"{name:<26}"
              f"{np.mean([r[0] for r in reached]):>10.0f}"
              f"{np.mean([r[1] for r in reached]):>10.2f}"
              f"{np.mean([r[2] for r in reached]):>10.2f}")

    # 「到達後に 0 にする」効果は，同じシードどうしを対にして差を見ないと分からない
    fixed  = np.array([table[("0.10 固定", seed)][2] for seed in SEEDS])
    switch = np.array([table[("0.10 → 到達後に 0.00", seed)][2] for seed in SEEDS])
    diff   = switch - fixed
    print("\n--- 0.10 固定 vs 0.10 → 到達後に 0.00（同じシードで対にした差）---")
    print(f"最終コストの差（切替 − 固定）＝ {diff.mean():+.2f} ± {diff.std(ddof=1):.2f}"
          f"（切替が勝った回数 {int((diff < 0).sum())}/{len(SEEDS)}）")
    print("→ 差はシードごとのばらつきに埋もれる（＝切り替えても最終コストは変わらない）")


if __name__ == "__main__":
    main()
