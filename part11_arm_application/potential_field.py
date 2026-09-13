# 経路生成手法であるポテンシャル法（potential field）の実装
# ポテンシャル法：ゴールへの「引力」と障害物からの「斥力」を足し合わせた力に沿って，
#                  ロボットを1歩ずつ動かして経路を作る手法（連続空間で動く・グリッド不要）

# ライブラリの読み込み
import numpy as np

# 自作モジュールの読み込み
from constant import *          # 定数
from grid_map import GridMap    # 占有格子地図


def obstacle_positions(grid_map: GridMap) -> np.ndarray:
    """
    占有格子地図から，障害物セルの中心座標 (x, y) を一覧で取り出す

    パラメータ
        grid_map: 占有格子地図

    戻り値
        positions: 障害物セルの中心座標を並べた配列（shape=(障害物数, 2)）
    """
    # grid が 1（障害物）のセルの (row, col) を取り出す
    rows, cols = np.where(grid_map.grid == CELL_OBSTACLE)
    # x が列(col)，y が行(row) に対応する（grid_map.cell_to_pos と同じ並び）
    positions = np.column_stack([cols, rows]).astype(float)
    return positions


def attractive_force(pos: np.ndarray, goal: np.ndarray,
                     k_att: float = K_ATTRACTIVE) -> np.ndarray:
    """
    引力 F_att を計算する（ゴール方向へ引く力）

    引力ポテンシャル U_att(q) = (1/2) k_att |q - q_goal|^2 の勾配の逆向き：
        F_att = -∇U_att = -k_att (q - q_goal)
    ゴールから遠いほど強く，ゴールにいると 0 になる。

    パラメータ
        pos: 現在位置 (x, y)
        goal: ゴール位置 (x, y)
        k_att: 引力ゲイン（大きいほど強く引かれる）

    戻り値
        force: 引力ベクトル (fx, fy)
    """
    pos  = np.asarray(pos,  dtype=float)
    goal = np.asarray(goal, dtype=float)
    return -k_att * (pos - goal)


def repulsive_force(pos: np.ndarray, obstacles: np.ndarray,
                    influence_radius: float = INFLUENCE_RADIUS,
                    k_rep: float = K_REPULSIVE) -> np.ndarray:
    """
    斥力 F_rep を計算する（障害物から離す力）

    影響半径 ρ0 内にある障害物だけが，距離 ρ が近いほど急増する斥力を出す：
        U_rep(q) = (1/2) k_rep (1/ρ - 1/ρ0)^2   （ρ ≤ ρ0 のときだけ。それ以外は 0）
        F_rep    = -∇U_rep = k_rep (1/ρ - 1/ρ0) (1/ρ^2) * (q - q_obs)/ρ
    各障害物からの斥力をすべて足し合わせる。

    パラメータ
        pos: 現在位置 (x, y)
        obstacles: 障害物の座標を並べた配列（shape=(障害物数, 2)）
        influence_radius: 斥力の影響半径 ρ0（これより遠い障害物は無視）
        k_rep: 斥力ゲイン（大きいほど強く押し返される）

    戻り値
        force: 斥力ベクトル (fx, fy)
    """
    pos   = np.asarray(pos, dtype=float)
    force = np.zeros(DIMENSION_2D, dtype=float)

    if len(obstacles) == 0:
        return force

    for obs in obstacles:
        # 障害物から自分へ向かうベクトルと，その距離 ρ
        difference = pos - obs
        rho = np.linalg.norm(difference)
        if rho < EPSILON:
            # 障害物のほぼ真上（0割を避けるため飛ばす。実際の点ロボットは入らない）
            continue
        if rho <= influence_radius:
            # 影響半径の内側だけ斥力が働く（ρ0 でちょうど 0 になめらかに消える）
            magnitude = k_rep * (1.0 / rho - 1.0 / influence_radius) / (rho ** 2)
            force += magnitude * (difference / rho)

    return force


def total_force(pos: np.ndarray, goal: np.ndarray, obstacles: np.ndarray,
                k_att: float = K_ATTRACTIVE, k_rep: float = K_REPULSIVE,
                influence_radius: float = INFLUENCE_RADIUS) -> np.ndarray:
    """
    引力と斥力を合成した合力 F = F_att + F_rep を計算する

    パラメータ
        pos: 現在位置 (x, y)
        goal: ゴール位置 (x, y)
        obstacles: 障害物の座標を並べた配列
        k_att: 引力ゲイン
        k_rep: 斥力ゲイン
        influence_radius: 斥力の影響半径 ρ0

    戻り値
        force: 合力ベクトル (fx, fy)
    """
    f_att = attractive_force(pos, goal, k_att)
    f_rep = repulsive_force(pos, obstacles, influence_radius, k_rep)
    return f_att + f_rep


def plan_potential_field(grid_map: GridMap,
                         k_att: float = K_ATTRACTIVE, k_rep: float = K_REPULSIVE,
                         influence_radius: float = INFLUENCE_RADIUS,
                         step_gain: float = STEP_GAIN,
                         max_step_length: float = MAX_STEP_LENGTH,
                         max_steps: int = MAX_STEPS,
                         goal_tolerance: float = GOAL_TOLERANCE,
                         patience: int = STUCK_PATIENCE) -> tuple:
    """
    ポテンシャル法で経路を生成する（合力に沿って1歩ずつ進む＝勾配降下）

    各位置で合力 F を求め，F の向きに少しだけ進む，を繰り返す。
    ゴールに十分近づけば成功。力が釣り合って動けなくなれば失敗（局所最小値に捕まる）。

    パラメータ
        grid_map: 占有格子地図（スタート・ゴール設定済み）
        k_att: 引力ゲイン
        k_rep: 斥力ゲイン
        influence_radius: 斥力の影響半径 ρ0
        step_gain: 力を移動量へ変換するゲイン
        max_step_length: 1ステップで進める最大距離（暴走防止）
        max_steps: 最大反復回数
        goal_tolerance: ゴール到達とみなす距離
        patience: ゴールへの最短距離がこの回数だけ更新されなければ局所最小値とみなす

    戻り値
        path: 経路（連続座標 (x, y) のリスト）
        reached: True / False = ゴールに到達できた / 局所最小値などで失敗した
    """
    # スタート・ゴールを連続座標へ変換し，障害物の座標一覧を用意する
    goal      = grid_map.cell_to_pos(grid_map.goal)
    pos       = grid_map.cell_to_pos(grid_map.start)
    obstacles = obstacle_positions(grid_map)

    path    = [pos.copy()]
    reached = False

    # 局所最小値の検出用：ゴールへの最短到達距離と，それが更新されない連続回数
    best_distance = np.linalg.norm(pos - goal)
    no_progress   = 0

    for _ in range(max_steps):
        distance = np.linalg.norm(pos - goal)
        # ゴールに十分近づいたら成功で終了
        if distance <= goal_tolerance:
            reached = True
            break

        # ゴールへの最短距離が縮んだかを見て，進捗がない回数を数える
        if distance < best_distance - EPSILON:
            best_distance = distance
            no_progress   = 0
        else:
            no_progress += 1
            if no_progress >= patience:
                # しばらくゴールへ近づけていない＝引力と斥力が釣り合い局所最小値に捕まった
                break

        # 合力を計算し，その向きへ1ステップ進む
        force = total_force(pos, goal, obstacles, k_att, k_rep, influence_radius)
        step  = step_gain * force

        # 1ステップの移動量が大きすぎる場合は頭打ちにする（引力が強い所での暴走防止）
        step_length = np.linalg.norm(step)
        if step_length > max_step_length:
            step = step / step_length * max_step_length
            step_length = max_step_length

        if step_length < STUCK_THRESHOLD:
            # ほとんど動けない＝引力と斥力が完全に釣り合って局所最小値に捕まった
            break

        pos = pos + step
        path.append(pos.copy())

    return path, reached
