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


def _draw_overlay(ax: Axes, grid_map: GridMap, path: list, explored: list) -> None:
    """
    探索済みセルと経路を地図の上に重ねて描く（内部用）

    パラメータ
        ax: 描画先の Axes
        grid_map: 占有格子地図
        path: 経路（セル (row, col) のリスト）
        explored: 探索済みセル（セル (row, col) のリスト）
    """
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
    _draw_overlay(ax, grid_map, path, explored)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(file_name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_compare(grid_map: GridMap, results: list, file_name: str,
                 suptitle: str = "") -> None:
    """
    複数手法の探索結果を横並びのパネルで比較して保存する（探索範囲の広がり方を見比べる）

    パラメータ
        grid_map: 占有格子地図
        results: [(手法名, path, explored), ...] のリスト
        file_name: 保存ファイル名
        suptitle: 図全体のタイトル
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2))
    if n == 1:
        axes = [axes]

    for ax, (name, path, explored) in zip(axes, results):
        _draw_overlay(ax, grid_map, path, explored)
        # パネル見出しに「手法名」と「展開ノード数」を入れて比較しやすくする
        ax.set_title(f"{name}\n(explored={len(explored)})", fontsize=11)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
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


# --- ここから第5章（ポテンシャル法）用：ベクトル場と連続経路の描画 ---

def _continuous_path_array(path: list) -> np.ndarray:
    """
    連続座標 (x, y) のリストを numpy 配列にまとめる（内部用）

    パラメータ
        path: 連続座標 (x, y) のリスト

    戻り値
        p: shape=(点数, 2) の配列（空なら shape=(0, 2)）
    """
    return np.array(path) if path else np.empty((0, 2))


def _draw_quiver(ax: Axes, grid_map: GridMap, force_func, grid_step: int, alpha: float) -> None:
    """
    位置ごとの力ベクトルを矢印（quiver）で重ね描きする（内部用）

    パラメータ
        ax: 描画先の Axes
        grid_map: 占有格子地図
        force_func: 位置 (x, y) を受け取り力ベクトル (fx, fy) を返す関数
        grid_step: 矢印を何セルおきに描くか（大きいほど矢印が粗くなる）
        alpha: 矢印の不透明度
    """
    rows, cols = grid_map.shape
    xs, ys, us, vs = [], [], [], []
    for row in range(0, rows, grid_step):
        for col in range(0, cols, grid_step):
            if grid_map.is_obstacle((row, col)):
                # 障害物セルの上には矢印を描かない
                continue
            fx, fy = force_func(np.array([float(col), float(row)]))
            norm = (fx ** 2 + fy ** 2) ** 0.5
            if norm < 1e-9:
                # 力がほぼ 0 の点（釣り合い）は矢印を描かない
                continue
            # 矢印は向きが分かれば十分なので，長さをそろえて見やすくする
            xs.append(col)
            ys.append(row)
            us.append(fx / norm)
            vs.append(fy / norm)    # y 軸は imshow(origin="upper") で反転済みなので，力の y 成分はそのまま渡す

    ax.quiver(xs, ys, us, vs, color="tab:blue", alpha=alpha,
              angles="xy", scale_units="xy", scale=1.6, width=0.004)


def plot_vector_field(grid_map: GridMap, force_func, file_name: str,
                      grid_step: int = 1, title: str = "") -> None:
    """
    位置ごとの力ベクトルを矢印（quiver）で描き，ポテンシャル場の様子を可視化する

    パラメータ
        grid_map: 占有格子地図
        force_func: 位置 (x, y) を受け取り力ベクトル (fx, fy) を返す関数
        file_name: 保存ファイル名
        grid_step: 矢印を何セルおきに描くか（大きいほど矢印が粗くなる）
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_base(ax, grid_map)
    _draw_quiver(ax, grid_map, force_func, grid_step, alpha=0.7)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(file_name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_potential_path(grid_map: GridMap, path: list, file_name: str,
                        reached: bool = True, force_func=None,
                        grid_step: int = 1, title: str = "") -> None:
    """
    ポテンシャル場（任意）の上に，ロボットがたどった連続経路を重ねて描く

    パラメータ
        grid_map: 占有格子地図
        path: 経路（連続座標 (x, y) のリスト）
        file_name: 保存ファイル名
        reached: ゴールに到達できたか（失敗時は経路色を変え，停止点を強調する）
        force_func: 力ベクトルを返す関数（指定すると背景にベクトル場も薄く描く）
        grid_step: ベクトル場の矢印を何セルおきに描くか
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_base(ax, grid_map)

    # 背景にベクトル場（任意・薄め）
    if force_func is not None:
        _draw_quiver(ax, grid_map, force_func, grid_step, alpha=0.3)

    # 連続経路（成功＝橙の実線 / 失敗＝赤の破線）
    p = _continuous_path_array(path)
    if len(p) > 0:
        if reached:
            ax.plot(p[:, 0], p[:, 1], "-", color="tab:orange", linewidth=2, label="path")
        else:
            ax.plot(p[:, 0], p[:, 1], "--", color="tab:red", linewidth=2, label="path (stuck)")
            # 動けなくなった停止点（局所最小値）を強調
            ax.plot(p[-1, 0], p[-1, 1], "X", color="tab:red", markersize=12,
                    label="local minimum")

    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(file_name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def animate_potential_path(grid_map: GridMap, path: list, file_name: str,
                           step: int = 2, title: str = "") -> None:
    """
    ロボットが力に沿って1歩ずつ進む様子を gif アニメーションで保存する

    パラメータ
        grid_map: 占有格子地図
        path: 経路（連続座標 (x, y) のリスト）
        file_name: 保存ファイル名
        step: 1フレームで進める点数（大きいほど短い gif になる）
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_base(ax, grid_map)
    if title:
        ax.set_title(title)

    # たどった軌跡（線）と現在地（点）を空で用意し，フレームごとに伸ばす
    trail_plot, = ax.plot([], [], "-", color="tab:orange", linewidth=2)
    robot_plot, = ax.plot([], [], "o", color="tab:orange", markersize=8)

    p = _continuous_path_array(path)
    n = len(p)
    # 最後のコマをくり返して，到達した姿を約1.5秒だけ見せてから先頭へ戻す（gif はループ再生のため）
    frames = list(range(1, n, step)) + [n] * _hold_frames(20)

    def update(k: int) -> tuple:
        trail_plot.set_data(p[:k, 0], p[:k, 1])
        if k > 0:
            robot_plot.set_data([p[k - 1, 0]], [p[k - 1, 1]])
        return trail_plot, robot_plot

    anim = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
    anim.save(file_name, writer="pillow", fps=20)
    plt.close(fig)
