# 関節空間（C-space）と作業空間の可視化（静止画 png）
#
# 本章の図は2種類：
#   ① 作業空間 → 関節空間（C-space）の対応を示す図（11a 理論編）
#   ② 8手法の探索のあと（禁止領域・探索の広がり・経路）を1枚に並べた比較図（11b 実装編）
# 経路や探索の点はすべて「セル座標 (x, y)」（x＝列＝θ1方向，y＝行＝θ2方向）で受け取り，
# 軸の目盛りだけ関節角度 [deg] に読み替えて表示する。

# ライブラリの読み込み
import numpy as np                      # 数値計算
import matplotlib.pyplot as plt         # 描画用
import matplotlib.patches as patches    # 2次元形状の描画
from matplotlib.axes import Axes         # 型ヒント用（描画先の Axes 型）

# 自作モジュールの読み込み
from constant import *                  # 定数
from grid_map import CSpaceMap          # 関節空間（C-space）地図


# C-space の軸目盛りに使う関節角度 [deg]（-180, -90, 0, 90, 180）
_DEG_TICKS = [-180, -90, 0, 90, 180]


def _draw_cspace_background(axis: Axes, cspace_map: CSpaceMap) -> None:
    """
    C-space地図の背景（禁止領域＝C-obstacle）と軸目盛り（関節角度）を描く（内部用）

    パラメータ
        axis: 描画先の軸
        cspace_map: 関節空間（C-space）地図
    """
    rows, cols = cspace_map.shape
    # 禁止領域（1）を灰色で塗る。origin="lower" で行0（θ2=−180°）を下に置く
    axis.imshow(cspace_map.grid, origin="lower", extent=[0, cols - 1, 0, rows - 1],
                cmap="Greys", alpha=0.85, vmin=0, vmax=1, aspect="equal")

    # 軸目盛りをセル座標から関節角度 [deg] へ読み替える
    def _deg_to_coord(deg, n):
        return (deg - np.rad2deg(JOINT_MIN)) / (np.rad2deg(JOINT_MAX) - np.rad2deg(JOINT_MIN)) * (n - 1)
    axis.set_xticks([_deg_to_coord(d, cols) for d in _DEG_TICKS])
    axis.set_xticklabels(_DEG_TICKS)
    axis.set_yticks([_deg_to_coord(d, rows) for d in _DEG_TICKS])
    axis.set_yticklabels(_DEG_TICKS)
    axis.set_xlabel("joint 1  θ1 [deg]")
    axis.set_ylabel("joint 2  θ2 [deg]")


def _draw_start_goal(axis: Axes, cspace_map: CSpaceMap) -> None:
    """
    C-space上のスタート（水色）・ゴール（赤）を描く（内部用）
    """
    start = cspace_map.cell_to_pos(cspace_map.start)
    goal  = cspace_map.cell_to_pos(cspace_map.goal)
    axis.scatter(start[0], start[1], color="cyan", edgecolors="black", s=70, zorder=6, label="start")
    axis.scatter(goal[0],  goal[1],  color="red",  edgecolors="black", s=70, zorder=6, label="goal")


def _draw_overlay(axis: Axes, result: dict) -> None:
    """
    探索の広がり（グラフ系は探索済みセル，サンプリング系は木・ロードマップの枝）を薄く描く（内部用）

    パラメータ
        axis: 描画先の軸
        result: 手法の結果（explored / edges / path を持つ辞書）
    """
    # グラフ系：確定した探索済みセルを薄い点で示す
    explored = result.get("explored")
    if explored:
        explored = np.array(explored)
        axis.scatter(explored[:, 0], explored[:, 1], color="tab:orange", s=4, alpha=0.25, zorder=2)

    # サンプリング系：木・ロードマップの枝を薄い線で示す
    edges = result.get("edges")
    if edges:
        for parent, child in edges:
            axis.plot([parent[0], child[0]], [parent[1], child[1]],
                      color="tab:green", linewidth=0.4, alpha=0.4, zorder=2)

    # 経路を太い線で重ねる（到達できていれば）
    path = result.get("path")
    if path:
        path = np.array(path)
        axis.plot(path[:, 0], path[:, 1], color="tab:blue", linewidth=2.0, zorder=4)


def plot_cspace_compare(cspace_map: CSpaceMap, results: list, file_name: str) -> None:
    """
    8手法の C-space 探索結果（禁止領域・探索の広がり・経路）を2×4で並べた比較図を作る

    パラメータ
        cspace_map: 関節空間（C-space）地図
        results: 各手法の結果辞書のリスト（name / path / edges / explored / reached / cost を持つ）
        file_name: 出力する png のファイル名
    """
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    figure, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for axis, result in zip(axes, results):
        _draw_cspace_background(axis, cspace_map)
        _draw_overlay(axis, result)
        _draw_start_goal(axis, cspace_map)
        # タイトルに到達可否とコストを添える
        if result.get("reached"):
            subtitle = f"cost={result['cost']:.1f}"
        else:
            subtitle = "not reached"
        axis.set_title(f"{result['name']}  ({subtitle})")

    # 余ったマス目（手法が8未満なら）を消す
    for axis in axes[n:]:
        axis.axis("off")

    figure.suptitle("8 path planners on the 2-DoF arm joint space (C-space)", fontsize=14)
    figure.tight_layout()
    figure.savefig(file_name, dpi=120)
    plt.close(figure)


def _draw_arm(axis: Axes, robot, joint: np.ndarray, color: str, label: str) -> None:
    """
    作業空間にある姿勢のアームを1つ描く（内部用）

    パラメータ
        axis: 描画先の軸
        robot: 2軸ロボットアーム
        joint: 関節角度 (θ1, θ2) [rad]
        color: 線の色
        label: 凡例ラベル
    """
    all_link_pos = robot.forward_kinematics_all_link_pos(joint)
    axis.plot(all_link_pos[:, 0], all_link_pos[:, 1], color=color, linewidth=3, marker="o", label=label)


def plot_workspace_to_cspace(cspace_map: CSpaceMap, file_name: str) -> None:
    """
    「作業空間 → 関節空間（C-space）」の対応を示す図を作る（11a 理論編）

    左：作業空間。障害物と，スタート姿勢・ゴール姿勢のアームを描く。
    右：関節空間（C-space）。同じ障害物が禁止領域（C-obstacle）へ姿を変え，
        スタート・ゴールが2つの点になることを示す。

    パラメータ
        cspace_map: 関節空間（C-space）地図
        file_name: 出力する png のファイル名
    """
    robot       = cspace_map.robot
    environment = cspace_map.environment
    limit = sum(ARM_LINK_LENGTHS) + 0.3

    start_joint = cspace_map.coord_to_joint(cspace_map.cell_to_pos(cspace_map.start))
    goal_joint  = cspace_map.coord_to_joint(cspace_map.cell_to_pos(cspace_map.goal))

    figure, (ax_work, ax_cspace) = plt.subplots(1, 2, figsize=(11, 5.2))

    # --- 左：作業空間 ---
    for name, datas in environment.interferences.items():
        if name == INTERFERENCE.CIRCLE:
            for x, y, radius in datas:
                ax_work.add_patch(patches.Circle((x, y), radius, color="gray", alpha=0.5))
        elif name == INTERFERENCE.RECTANGLE:
            for x, y, center_x, center_y, angle in datas:
                ax_work.add_patch(patches.Rectangle((center_x - x / 2, center_y - y / 2),
                                                    x, y, angle=angle, color="gray", alpha=0.5))
    _draw_arm(ax_work, robot, start_joint, "cyan", "start pose")
    _draw_arm(ax_work, robot, goal_joint,  "red",  "goal pose")
    ax_work.set_xlim(-limit, limit)
    ax_work.set_ylim(-limit, limit)
    ax_work.set_aspect("equal")
    ax_work.grid()
    ax_work.set_xlabel("X [m]")
    ax_work.set_ylabel("Y [m]")
    ax_work.set_title("workspace (arm + obstacles)")
    ax_work.legend(loc="upper left", fontsize=8)

    # --- 右：関節空間（C-space）---
    _draw_cspace_background(ax_cspace, cspace_map)
    _draw_start_goal(ax_cspace, cspace_map)
    ax_cspace.set_title("joint space (C-space): obstacles become forbidden regions")
    ax_cspace.legend(loc="upper right", fontsize=8)

    figure.suptitle("workspace obstacle -> C-obstacle (forbidden region in joint space)", fontsize=13)
    figure.tight_layout()
    figure.savefig(file_name, dpi=120)
    plt.close(figure)
