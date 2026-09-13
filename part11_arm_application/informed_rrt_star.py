# 経路生成手法である Informed RRT*（インフォームドRRTスター：RRT* に「初期解が見つかった後は
#   サンプリングする領域を楕円体〔ellipsoid：だ円を回転させたような立体。2次元では「だ円」〕に
#   絞り込む」工夫を足したアルゴリズム）の実装
# Informed RRT*（初期解が出るまでは RRT* と同じく地図全体を一様サンプリング。初期解が出たあとは，
#       スタートとゴールを焦点とし，現在の最良経路長 c_best を長径とする楕円体の内側だけをサンプルする。
#       この楕円体の外側にある点は，たとえ経路に使っても c_best より長くなるので最適化には無関係＝捨ててよい。
#       c_best が縮むたびに楕円体も細くなり，サンプルが最適経路の周りへ集中して収束が速くなる）

# ライブラリの読み込み
import numpy as np    # 数値計算

# 自作モジュールの読み込み
from constant import *                              # 定数
from grid_map import GridMap                        # 占有格子地図
from collision import is_collision_segment          # 線分の衝突判定
# 第7章 RRT の部品（一様サンプリング・ステップ伸長）をそのまま再利用する
from rrt import sample, steer
# 第9章 RRT* の部品（コスト付き木・近傍半径・最良の親選び・つなぎ直し）をそのまま再利用する
from rrt_star import StarTree, neighbor_radius, choose_parent, rewire


def rotation_to_world_frame(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """
    楕円体の「自分の座標系」から「地図の座標系（ワールド座標）」へ向きを合わせる回転行列 C を求める

    楕円体は「スタートとゴールを結ぶ向き」を長軸（横方向）として作る。この長軸を地図の x 軸に
    そろえる回転行列を，特異値分解（SVD：行列を回転・拡大・回転の3つに分解する定番手法）で組み立てる。
    こうすると，まっすぐな単位円から作った点を，正しい向きの楕円体へ回せる。

    パラメータ
        start: スタートの連続座標 (x, y)（楕円体の片方の焦点）
        goal: ゴールの連続座標 (x, y)（楕円体のもう片方の焦点）

    戻り値
        C: 楕円体座標系 → ワールド座標系 への回転行列（2×2）
    """
    # 長軸方向の単位ベクトル a1（スタート → ゴール の向き）
    difference = np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)
    a1 = difference / np.linalg.norm(difference)

    # a1 を地図の x 軸 e1=(1,0) へ対応づける行列 M = a1・e1^T を作り，SVD で回転成分を取り出す
    e1 = np.zeros(DIMENSION_2D)
    e1[0] = 1.0
    M = np.outer(a1, e1)
    U, _S, Vt = np.linalg.svd(M)

    # det(U)・det(V) を中央成分に入れることで，鏡映（裏返し）にならない純粋な回転にそろえる
    middle = np.diag([1.0, np.linalg.det(U) * np.linalg.det(Vt)])
    C = U @ middle @ Vt
    return C


def sample_unit_ball(rng: np.random.Generator) -> np.ndarray:
    """
    半径1の円（単位球の2次元版）の内側から一様に1点をサンプルする

    単純に x,y を一様に取ると中心に偏るので，半径は √u（u は一様乱数）で取り，
    角度は 0〜2π で一様に取る（こうすると円内に均等にばらまける）。

    パラメータ
        rng: 乱数生成器（シードを固定して再現性を持たせる）

    戻り値
        point: 単位円内の点 (x, y)
    """
    radius = np.sqrt(rng.random())          # 面積が均等になるよう √ を取る
    angle  = rng.uniform(0.0, 2.0 * np.pi)  # 角度は一様
    return np.array([radius * np.cos(angle), radius * np.sin(angle)])


def sample_informed(c_best: float, c_min: float, center: np.ndarray,
                    C: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    スタート・ゴールを焦点とする楕円体の内側から一様に1点をサンプルする（Informed サンプリング）

    手順は「単位円の点 → 楕円の大きさへ引き伸ばす（対角行列 L）→ 正しい向きへ回す（回転行列 C）→
    中心（2焦点の中点）へ運ぶ」の4ステップ。長径は c_best/2，短径は √(c_best² − c_min²)/2 で，
    c_best が c_min に近づくほど楕円が細くなる（最適経路の周りへサンプルが集中する）。

    パラメータ
        c_best: 現在の最良経路長（楕円の長径＝2焦点からの距離の和の上限）
        c_min: 2焦点間の距離（スタート〜ゴールの直線距離。これより短い経路はありえない）
        center: 楕円の中心（スタートとゴールの中点）
        C: 楕円体座標系 → ワールド座標系 への回転行列
        rng: 乱数生成器（シードを固定して再現性を持たせる）

    戻り値
        point: 楕円体内の連続座標 (x, y)
    """
    # 長軸方向の半径 r1 と，それ以外の軸の半径 ri（2次元なので短軸が1つ）
    r1 = c_best / 2.0
    ri = np.sqrt(max(c_best ** 2 - c_min ** 2, 0.0)) / 2.0
    L  = np.diag([r1, ri])

    # 単位円の点を L で引き伸ばし，C で回し，center へ平行移動する
    x_ball = sample_unit_ball(rng)
    point  = C @ L @ x_ball + center
    return point


def plan_informed_rrt_star(grid_map: GridMap,
                           max_iterations: int = INFORMED_MAX_ITERATIONS,
                           step_size: float = INFORMED_STEP_SIZE,
                           goal_sample_rate: float = INFORMED_GOAL_SAMPLE_RATE,
                           goal_tolerance: float = INFORMED_GOAL_TOLERANCE,
                           segment_step: float = INFORMED_SEGMENT_STEP,
                           gamma: float = INFORMED_NEIGHBOR_GAMMA,
                           max_radius: float = INFORMED_NEIGHBOR_MAX,
                           seed: int = INFORMED_RANDOM_SEED) -> tuple:
    """
    Informed RRT* でスタートからゴールまでの経路を生成する

    RRT* と同じく choose parent（最良の親選び）と rewire（つなぎ直し）で経路を最適化するが，
    サンプリングだけが違う。初期解が見つかるまでは地図全体を一様にサンプルし，初期解が出た後は
    スタート・ゴールを焦点とする楕円体の内側だけをサンプルする。c_best が縮むたびに楕円も細くなる。

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
        tree: 成長させた Informed RRT* 探索木
        cost_history: (試行回数, その時点の最良コスト) のリスト（収束カーブ用）
        snapshots: 最良経路が更新されるたびの (試行回数, コスト, 経路, 楕円パラメータ) のリスト（gif用）
        informed_samples: 楕円体内からサンプルした点のリスト（サンプルの集中ぐあいの可視化用）
    """
    rng = np.random.default_rng(seed)

    start_pos = grid_map.cell_to_pos(grid_map.start)
    goal_pos  = grid_map.cell_to_pos(grid_map.goal)

    # 楕円体の「固定パラメータ」（c_best 以外）は最初に1度だけ求めておく
    c_min  = float(np.linalg.norm(goal_pos - start_pos))   # 2焦点間の距離（最短ありえない経路長）
    center = (start_pos + goal_pos) / 2.0                  # 楕円の中心（2焦点の中点）
    C      = rotation_to_world_frame(start_pos, goal_pos)  # 楕円体 → ワールド の回転行列

    # スタートを根として RRT* 木を初期化
    tree = StarTree(start_pos)

    goal_candidates  = []          # ゴールへ直進できるノード番号（ここから最良経路を選ぶ）
    cost_history     = []          # (試行回数, 最良コスト) … 反復ごとのコスト低下を記録
    snapshots        = []          # (試行回数, コスト, 経路, 楕円) … 最良経路が更新されたとき記録
    informed_samples = []          # 楕円体内からサンプルした点（可視化用）
    best_cost = float("inf")

    for i in range(1, max_iterations + 1):
        # ステップ1：サンプリング。初期解が出る前は一様，出た後は楕円体内から取る（ここが RRT* との違い）
        if best_cost == float("inf"):
            # まだ経路が見つかっていない → 地図全体を一様にサンプル（ゴールバイアスつき）
            rand_point = sample(grid_map, goal_pos, goal_sample_rate, rng)
        else:
            # 経路が見つかった → 楕円体の内側だけをサンプルし，無駄を省く
            #   ただし楕円が地図からはみ出すこともあるので，自由セルに入るまで引き直す（上限つき）
            rand_point = None
            for _ in range(20):
                candidate = sample_informed(best_cost, c_min, center, C, rng)
                cell = grid_map.pos_to_cell(candidate)
                if grid_map.is_free(cell):
                    rand_point = candidate
                    break
            if rand_point is None:
                # うまく取れなければ一様サンプルへフォールバック（試行を無駄にしない）
                rand_point = sample(grid_map, goal_pos, goal_sample_rate, rng)
            else:
                informed_samples.append(rand_point)

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
                # 最良コストが更新された → 経路と，その c_best に対応する楕円を1枚スナップショット（gif用）
                best_cost = current_best
                best_idx  = goal_candidates[int(np.argmin(costs))]
                path = tree.path_to(best_idx) + [goal_pos]
                ellipse = (center, C, best_cost, c_min)
                snapshots.append((i, current_best, path, ellipse))

    # 最終的な最良経路を組み立てる（候補がなければ経路なし）
    if goal_candidates:
        costs = [tree.cost(idx) + float(np.linalg.norm(tree.nodes[idx] - goal_pos))
                 for idx in goal_candidates]
        best_idx = goal_candidates[int(np.argmin(costs))]
        path = tree.path_to(best_idx) + [goal_pos]
    else:
        path = []

    return path, tree, cost_history, snapshots, informed_samples


if __name__ == "__main__":
    # 動作確認（共通サンプル地図で Informed RRT* を回し，楕円体サンプリングが効くか確かめる）
    from grid_map import make_sample_map

    grid_map = make_sample_map()
    path, tree, cost_history, snapshots, informed_samples = plan_informed_rrt_star(grid_map)
    print(f"木のノード数 = {tree.n_nodes} / 最良経路の更新回数 = {len(snapshots)}")
    print(f"楕円体内からサンプルした回数 = {len(informed_samples)}")
    if path:
        first_cost = snapshots[0][1]
        last_cost  = snapshots[-1][1]
        print(f"到達 = True / 経路の通過点数 = {len(path)}")
        print(f"初期解コスト = {first_cost:.2f} → 最終コスト = {last_cost:.2f}"
              f"（{first_cost - last_cost:.2f} 短縮）")
    else:
        print("到達 = False（最大試行回数までゴールへ届かなかった）")
