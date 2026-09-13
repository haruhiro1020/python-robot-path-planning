# 経路生成手法である RRT-Connect（双方向RRT：スタート側とゴール側の2本の木を
#   交互に伸ばし，中央で出会わせて速くつなぐアルゴリズム）の実装
# RRT-Connect（スタート木 T_a とゴール木 T_b を持ち，T_a を1歩伸ばした新ノードへ
#               T_b を「衝突するまで一気に伸ばす〔CONNECT〕」で近づける。役割を交互に
#               入れ替え〔swap〕ながら，2本が接続できたら両木の経路を1本に結合する手法）

# ライブラリの読み込み
import numpy as np    # 数値計算

# 自作モジュールの読み込み
from constant import *                              # 定数
from grid_map import GridMap                        # 占有格子地図
from collision import is_collision_segment          # 線分の衝突判定
# 第7章 RRT の部品をそのまま再利用する（木構造・一様サンプリング・ステップ伸長）
from rrt import Tree, sample, steer


def extend(grid_map: GridMap, tree: Tree, target: np.ndarray,
           step_size: float, segment_step: float, events: list, label: str) -> tuple:
    """
    木を target の向きへ「1ステップだけ」伸ばす（RRTの1回ぶんと同じ）

    最近傍ノードから target 方向へ step_size だけ枝を伸ばし，衝突しなければ木に加える。
    伸ばした結果を 3 つの状態（REACHED / ADVANCED / TRAPPED）で返し，
    CONNECT 側がこの状態を見て「まだ伸ばすか」を決める。

    パラメータ
        grid_map: 占有格子地図
        tree: 伸ばす対象の探索木
        target: 伸ばしたい目標点の連続座標 (x, y)
        step_size: 1回で伸ばす距離（ステップ幅）
        segment_step: 枝（線分）の衝突判定の刻み幅
        events: 成長の様子（gif用）を記録するリスト（破壊的に追記する）
        label: どちらの木か（"start" / "goal"）。events と描画の色分けに使う

    戻り値
        status: 伸長結果（EXTEND_STATUS.REACHED / EXTEND_STATUS.ADVANCED / EXTEND_STATUS.TRAPPED）
        new_index: 追加したノード番号（TRAPPED のときは -1）
    """
    # 木の中で target にもっとも近いノードから枝を伸ばす
    nearest_index = tree.nearest(target)
    nearest_point = tree.nodes[nearest_index]
    new_point     = steer(nearest_point, target, step_size)

    # 枝（最近傍→新ノード）が障害物をまたぐなら1歩も伸ばせない（行き止まり）
    if is_collision_segment(grid_map, nearest_point, new_point, segment_step):
        return EXTEND_STATUS.TRAPPED, -1

    # 衝突しなければ木に追加し，成長の様子を記録する
    new_index = tree.add_node(new_point, nearest_index)
    events.append((label, (nearest_point, new_point)))

    # target そのものまで届いたか（steer は近ければ target に一致させる）で状態を分ける
    if np.allclose(new_point, target):
        return EXTEND_STATUS.REACHED, new_index
    return EXTEND_STATUS.ADVANCED, new_index


def connect(grid_map: GridMap, tree: Tree, target: np.ndarray,
            step_size: float, segment_step: float, events: list, label: str) -> tuple:
    """
    木を target へ「衝突するまで一気に」伸ばす（CONNECT 操作）

    通常RRTは1ステップで止まるが，CONNECT は REACHED か TRAPPED になるまで
    extend を繰り返す。空いている方向ならぐんぐん target に近づけるので速い。

    パラメータ
        grid_map: 占有格子地図
        tree: 伸ばす対象の探索木
        target: もう一方の木の新ノード（ここへ接続を試みる）
        step_size: 1回で伸ばす距離（ステップ幅）
        segment_step: 枝（線分）の衝突判定の刻み幅
        events: 成長の様子（gif用）を記録するリスト
        label: どちらの木か（"start" / "goal"）

    戻り値
        status: 最終状態（EXTEND_STATUS.REACHED＝接続成功 / EXTEND_STATUS.TRAPPED＝行き止まり）
        last_index: 最後に追加したノード番号（接続点。TRAPPED のときは -1）
    """
    last_index = -1
    while True:
        status, index = extend(grid_map, tree, target, step_size, segment_step, events, label)
        if status != EXTEND_STATUS.ADVANCED:
            # REACHED（接続成功）か TRAPPED（行き止まり）で打ち切る
            return status, last_index if status == EXTEND_STATUS.TRAPPED else index
        # まだ進める途中（ADVANCED）なら，今のノードを覚えて次の一歩へ
        last_index = index


def _merge_path(tree_start: Tree, index_start: int,
                tree_goal: Tree, index_goal: int) -> list:
    """
    接続点でつながった2本の木を，スタート→ゴールの1本の経路に結合する（内部用）

    パラメータ
        tree_start: スタートを根とする木
        index_start: スタート木側の接続点ノード番号
        tree_goal: ゴールを根とする木
        index_goal: ゴール木側の接続点ノード番号

    戻り値
        path: 経路（連続座標 (x, y) のリスト。start → … → goal の順）
    """
    # スタート木：根（start）→接続点 の順
    path_from_start = tree_start.path_to(index_start)
    # ゴール木：根（goal）→接続点 の順。反転して 接続点→goal にする
    path_from_goal  = tree_goal.path_to(index_goal)
    path_from_goal.reverse()
    # 接続点は両者で重複するので，ゴール側の先頭1点を落としてつなぐ
    return path_from_start + path_from_goal[1:]


def plan_rrt_connect(grid_map: GridMap,
                     max_iterations: int = RRT_CONNECT_MAX_ITERATIONS,
                     step_size: float = RRT_CONNECT_STEP_SIZE,
                     segment_step: float = RRT_CONNECT_SEGMENT_STEP,
                     seed: int = RRT_CONNECT_RANDOM_SEED) -> tuple:
    """
    RRT-Connect でスタートからゴールまでの経路を生成する

    スタート木 T_a を1ステップ伸ばし，その新ノードへゴール木 T_b を CONNECT する。
    つながらなければ2本の役割を入れ替え（swap）て繰り返す。
    両木が接続できたら，2本を1本の経路に結合して返す。

    パラメータ
        grid_map: 占有格子地図（スタート・ゴール設定済み）
        max_iterations: 試行の最大回数（swap1回を1反復と数える）
        step_size: 1回で伸ばす距離（ステップ幅）
        segment_step: 枝（線分）の衝突判定の刻み幅
        seed: 乱数シード（同じ結果を再現するために固定する）

    戻り値
        path: 経路（連続座標 (x, y) のリスト。到達できなければ空リスト）
        tree_start: スタートを根とする木
        tree_goal: ゴールを根とする木
        events: 成長の様子（gif用。(label, (親座標, 子座標)) を追加順に並べたリスト）
        n_iterations: 接続までに要した反復回数（失敗時は max_iterations）
    """
    rng = np.random.default_rng(seed)

    start_pos = grid_map.cell_to_pos(grid_map.start)
    goal_pos  = grid_map.cell_to_pos(grid_map.goal)

    # スタート木とゴール木を別々の根で初期化する
    tree_start = Tree(start_pos)
    tree_goal  = Tree(goal_pos)

    # 交互に伸ばすための作業用の参照（毎反復 swap して役割を入れ替える）
    tree_a, label_a = tree_start, "start"
    tree_b, label_b = tree_goal,  "goal"

    events = []   # gif 用：どちらの木にどの枝が生えたかを時系列で記録

    for i in range(1, max_iterations + 1):
        # ステップ1：自由空間から一様に1点サンプル（ゴールバイアスは使わない＝rate 0）
        rand_point = sample(grid_map, goal_pos, 0.0, rng)

        # ステップ2：木A を新サンプルの方向へ1ステップ伸ばす
        status_a, index_a = extend(grid_map, tree_a, rand_point,
                                   step_size, segment_step, events, label_a)
        if status_a != EXTEND_STATUS.TRAPPED:
            # ステップ3：A の新ノードへ，木B を衝突するまで一気に伸ばして接続を試みる
            new_point_a = tree_a.nodes[index_a]
            status_b, index_b = connect(grid_map, tree_b, new_point_a,
                                        step_size, segment_step, events, label_b)
            if status_b == EXTEND_STATUS.REACHED:
                # 2本が出会った。どちらが start 木かを見て経路を結合する
                if label_a == "start":
                    path = _merge_path(tree_a, index_a, tree_b, index_b)
                else:
                    path = _merge_path(tree_b, index_b, tree_a, index_a)
                return path, tree_start, tree_goal, events, i

        # ステップ4：つながらなければ A と B の役割を入れ替えて次の反復へ
        tree_a, tree_b   = tree_b, tree_a
        label_a, label_b = label_b, label_a

    # 最大反復回数まで2本が出会えなかった（経路なし）
    return [], tree_start, tree_goal, events, max_iterations


if __name__ == "__main__":
    # 動作確認（共通サンプル地図で2本の木を伸ばし，中央で接続できるか確かめる）
    from grid_map import make_sample_map

    grid_map = make_sample_map()
    path, tree_start, tree_goal, events, n_iterations = plan_rrt_connect(grid_map)
    n_nodes = tree_start.n_nodes + tree_goal.n_nodes
    print(f"反復回数 = {n_iterations} / 2木の合計ノード数 = {n_nodes}")
    if path:
        print(f"到達 = True / 経路の通過点数 = {len(path)}")
    else:
        print("到達 = False（最大反復回数まで2本の木が出会わなかった）")
