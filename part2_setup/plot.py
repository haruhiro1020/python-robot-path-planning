# 地図・経路・探索の様子を Matplotlib で描く共通部品（全手法で使い回す）

# ライブラリの読み込み
import numpy as np                          # 数値計算
import matplotlib.pyplot as plt             # 描画
from matplotlib.animation import FuncAnimation  # アニメーション
from matplotlib.axes import Axes         # 型ヒント用（描画先の Axes 型）

# 自作モジュールの読み込み
from constant import *          # 定数
from grid_map import GridMap    # 占有格子地図


# gif はくり返し再生されるので，ゴールに着いたコマをしばらく見せてから先頭へ戻す。
# すぐ先頭へ戻ると「速すぎて何が起きたか分からない」ため，最後のコマを何度も足して間を作る。
_GOAL_HOLD_SEC = 1.5   # ゴール到達後に静止させる時間（秒）


def _hold_frames(fps: int) -> int:
    """
    ゴール到達後に最後のコマを見せ続けるコマ数を，保存時の fps から求める

    パラメータ
        fps: gif の1秒あたりのコマ数（anim.save(..., fps=...) に渡す値）

    戻り値
        くり返すコマ数（最低1コマ）
    """
    return max(round(_GOAL_HOLD_SEC * fps), 1)


def _draw_base(ax: Axes, grid_map: GridMap) -> None:
    """
    地図・スタート・ゴールという「土台」を描く（内部用）

    パラメータ
        ax: 描画先の Axes
        grid_map: 占有格子地図
    """
    # 占有格子を白黒で描く（0=空き=白 / 1=障害物=黒）
    ax.imshow(grid_map.grid, cmap="Greys", origin="upper")

    # スタート（緑丸）とゴール（赤星）。plot は (x, y) = (列, 行) の順で指定する
    if grid_map.start is not None:
        ax.plot(grid_map.start[1], grid_map.start[0], "o",
                color="tab:green", markersize=10, label="start")
    if grid_map.goal is not None:
        ax.plot(grid_map.goal[1], grid_map.goal[0], "*",
                color="tab:red", markersize=14, label="goal")

    ax.set_xticks([])
    ax.set_yticks([])


def plot_map(grid_map: GridMap, file_name: str, title: str = "") -> None:
    """
    地図（占有格子）だけを描いて保存する

    パラメータ
        grid_map: 占有格子地図
        file_name: 保存ファイル名
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_base(ax, grid_map)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(file_name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_result(grid_map: GridMap, path: list, explored: list,
                file_name: str, title: str = "") -> None:
    """
    探索済みセルと経路を地図に重ねて描き，静止画として保存する

    パラメータ
        grid_map: 占有格子地図
        path: 経路（セル (row, col) のリスト）
        explored: 探索済みセル（セル (row, col) のリスト）
        file_name: 保存ファイル名
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_base(ax, grid_map)

    # 探索済みセル（水色の小さな四角）
    if explored:
        ex = np.array(explored)
        ax.plot(ex[:, 1], ex[:, 0], "s", color="tab:cyan",
                markersize=4, alpha=0.5, label="explored")

    # 経路（橙色の折れ線）
    if path:
        p = np.array(path)
        ax.plot(p[:, 1], p[:, 0], "-", color="tab:orange",
                linewidth=2, label="path")

    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(file_name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_search_animation(grid_map: GridMap, path: list, explored: list,
                          file_name: str, step: int = 10, title: str = "") -> None:
    """
    探索フロントが広がり，最後に経路が引かれる様子を gif アニメーションで保存する

    パラメータ
        grid_map: 占有格子地図
        path: 経路（セル (row, col) のリスト）
        explored: 探索済みセル（確定した順に並んだリスト）
        file_name: 保存ファイル名
        step: 1フレームで増やす探索セル数（大きいほど短い gif になる）
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_base(ax, grid_map)
    if title:
        ax.set_title(title)

    # 探索済みセルと経路を「空」で用意し，フレームごとに中身を更新する
    explored_plot, = ax.plot([], [], "s", color="tab:cyan", markersize=4, alpha=0.5)
    path_plot,     = ax.plot([], [], "-", color="tab:orange", linewidth=2)

    ex = np.array(explored) if explored else np.empty((0, 2))
    n  = len(explored)
    # 0, step, 2*step, ... と増やし，最後に全体＋経路を表示するフレームを足す
    # 最後のコマをくり返して，引けた経路を約1.5秒だけ見せてから先頭へ戻す（gif はループ再生のため）
    frames = list(range(0, n, step)) + [n] * _hold_frames(15)

    def update(k: int) -> tuple:
        # k 個目までの探索済みセルを表示
        explored_plot.set_data(ex[:k, 1], ex[:k, 0])
        if k >= n and path:
            # 探索が終わったフレームで経路を描く
            p = np.array(path)
            path_plot.set_data(p[:, 1], p[:, 0])
        return explored_plot, path_plot

    anim = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    anim.save(file_name, writer="pillow", fps=15)
    plt.close(fig)
