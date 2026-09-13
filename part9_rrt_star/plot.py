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
            vs.append(-fy / norm)   # imshow は上が row 小なので y 成分を反転

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


# --- ここから第6章（PRM）用：ロードマップ（点と辺のネットワーク）の描画 ---

def _draw_roadmap(ax: Axes, roadmap, node_color: str = "tab:blue",
                  edge_alpha: float = 0.3) -> None:
    """
    ロードマップ（ノード＝点，エッジ＝辺）を地図の上に重ねて描く（内部用）

    パラメータ
        ax: 描画先の Axes
        roadmap: ロードマップ（nodes / edge_list() を持つ）
        node_color: ノード（点）の色
        edge_alpha: エッジ（辺）の不透明度
    """
    # 辺（薄い灰色の線）を先に描いてから，点を上に重ねる
    for (x1, y1), (x2, y2) in roadmap.edge_list():
        ax.plot([x1, x2], [y1, y2], "-", color="gray",
                linewidth=0.6, alpha=edge_alpha, zorder=1)

    nodes = roadmap.nodes
    ax.plot(nodes[:, 0], nodes[:, 1], "o", color=node_color,
            markersize=3, zorder=2, label="roadmap")


def plot_roadmap(grid_map: GridMap, roadmap, file_name: str, title: str = "") -> None:
    """
    ロードマップ（撒いた点＋接続された辺）だけを地図に重ねて描いて保存する

    パラメータ
        grid_map: 占有格子地図
        roadmap: ロードマップ
        file_name: 保存ファイル名
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_base(ax, grid_map)
    _draw_roadmap(ax, roadmap)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(file_name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_prm_path(grid_map: GridMap, roadmap, path: list,
                  file_name: str, title: str = "") -> None:
    """
    ロードマップ（薄め）の上に，引かれた経路を重ねて描いて保存する

    パラメータ
        grid_map: 占有格子地図
        roadmap: ロードマップ
        path: 経路（連続座標 (x, y) のリスト）
        file_name: 保存ファイル名
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_base(ax, grid_map)
    # 背景にロードマップを薄く描く
    _draw_roadmap(ax, roadmap, edge_alpha=0.2)

    # 経路（橙色の太い折れ線）
    if path:
        p = np.array(path)
        ax.plot(p[:, 0], p[:, 1], "-", color="tab:orange",
                linewidth=2.5, zorder=3, label="path")

    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(file_name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def animate_prm(grid_map: GridMap, roadmap, path: list, n_samples: int,
                file_name: str, title: str = "") -> None:
    """
    「点を撒く → 辺でつなぐ → 経路を引く」の3フェーズを gif アニメーションで保存する

    パラメータ
        grid_map: 占有格子地図
        roadmap: ロードマップ（スタート・ゴール接続済み）
        path: 経路（連続座標 (x, y) のリスト）
        n_samples: 最初に撒いたサンプル点の数（この数だけ点をパラパラ表示する）
        file_name: 保存ファイル名
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_base(ax, grid_map)
    if title:
        ax.set_title(title)

    nodes    = roadmap.nodes
    segments = roadmap.edge_list()

    # ノード・エッジ・経路を「空」で用意し，フェーズごとに中身を増やす
    node_plot, = ax.plot([], [], "o", color="tab:blue", markersize=3, zorder=2)
    edge_lines = [ax.plot([], [], "-", color="gray", linewidth=0.6,
                          alpha=0.3, zorder=1)[0] for _ in segments]
    path_plot, = ax.plot([], [], "-", color="tab:orange", linewidth=2.5, zorder=3)

    # フェーズ1：点をパラパラ撒く / フェーズ2：辺を順につなぐ / フェーズ3：経路を引く
    sample_frames = list(range(0, n_samples + 1, max(n_samples // 30, 1)))
    edge_frames   = list(range(0, len(segments) + 1, max(len(segments) // 40, 1)))
    phase1 = [("sample", k) for k in sample_frames]
    phase2 = [("edge",   k) for k in edge_frames]
    phase3 = [("path",   1)] * _hold_frames(15)   # 引けた経路を約1.5秒だけ見せてから先頭へ戻す
    frames = phase1 + phase2 + phase3

    def update(frame: tuple) -> tuple:
        phase, k = frame
        if phase == "sample":
            # k 個目までのノードを表示（辺・経路はまだ）
            node_plot.set_data(nodes[:k, 0], nodes[:k, 1])
        elif phase == "edge":
            # 全ノードを出したうえで，k 本目までの辺を表示
            node_plot.set_data(nodes[:, 0], nodes[:, 1])
            for idx, line in enumerate(edge_lines):
                if idx < k:
                    (x1, y1), (x2, y2) = segments[idx]
                    line.set_data([x1, x2], [y1, y2])
        else:
            # 経路を引く
            if path:
                p = np.array(path)
                path_plot.set_data(p[:, 0], p[:, 1])
        return [node_plot, path_plot] + edge_lines

    anim = FuncAnimation(fig, update, frames=frames, interval=120, blit=False)
    anim.save(file_name, writer="pillow", fps=15)
    plt.close(fig)


# --- ここから第7章（RRT）用：探索木（ツリー）の描画 ---

def _draw_tree(ax: Axes, tree, edge_color: str = "tab:blue", edge_alpha: float = 0.5) -> None:
    """
    探索木（ノード＝点，枝＝親子をつなぐ線）を地図の上に重ねて描く（内部用）

    パラメータ
        ax: 描画先の Axes
        tree: 探索木（nodes / edge_list() を持つ）
        edge_color: 枝（線）の色
        edge_alpha: 枝（線）の不透明度
    """
    # 枝（親→子の線分）を先に描いてから，ノードを上に重ねる
    for (x1, y1), (x2, y2) in tree.edge_list():
        ax.plot([x1, x2], [y1, y2], "-", color=edge_color,
                linewidth=0.6, alpha=edge_alpha, zorder=1)

    nodes = tree.nodes
    ax.plot(nodes[:, 0], nodes[:, 1], "o", color=edge_color,
            markersize=2, zorder=2, label="tree")


def plot_rrt_tree(grid_map: GridMap, tree, file_name: str, title: str = "") -> None:
    """
    成長させた探索木（ノード＋枝）だけを地図に重ねて描いて保存する

    パラメータ
        grid_map: 占有格子地図
        tree: 探索木
        file_name: 保存ファイル名
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_base(ax, grid_map)
    _draw_tree(ax, tree)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(file_name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_rrt_path(grid_map: GridMap, tree, path: list,
                  file_name: str, title: str = "") -> None:
    """
    探索木（薄め）の上に，復元した経路を重ねて描いて保存する

    パラメータ
        grid_map: 占有格子地図
        tree: 探索木
        path: 経路（連続座標 (x, y) のリスト）
        file_name: 保存ファイル名
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_base(ax, grid_map)
    # 背景に探索木を薄く描く
    _draw_tree(ax, tree, edge_alpha=0.25)

    # 経路（橙色の太い折れ線）
    if path:
        p = np.array(path)
        ax.plot(p[:, 0], p[:, 1], "-", color="tab:orange",
                linewidth=2.5, zorder=3, label="path")

    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(file_name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def animate_rrt(grid_map: GridMap, tree, path: list,
                file_name: str, step: int = 8, title: str = "") -> None:
    """
    木が枝を1本ずつ伸ばして広がり，最後にゴールへ届いた経路を引く様子を gif で保存する

    パラメータ
        grid_map: 占有格子地図
        tree: 成長させた探索木
        path: 経路（連続座標 (x, y) のリスト）
        file_name: 保存ファイル名
        step: 1フレームで増やす枝の本数（大きいほど短い gif になる）
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_base(ax, grid_map)
    if title:
        ax.set_title(title)

    # 枝を追加順に並べておき，フレームごとに本数を増やしていく
    segments = tree.edge_list()
    edge_lines = [ax.plot([], [], "-", color="tab:blue", linewidth=0.6,
                          alpha=0.5, zorder=1)[0] for _ in segments]
    path_plot, = ax.plot([], [], "-", color="tab:orange", linewidth=2.5, zorder=3)

    n = len(segments)
    # 0, step, 2*step, ... と枝を増やし，最後に経路を見せるフレームを足す
    grow_frames = list(range(0, n + 1, step))
    # 最後のコマをくり返して，引けた経路を約1.5秒だけ見せてから先頭へ戻す（gif はループ再生のため）
    frames = [("grow", k) for k in grow_frames] + [("path", n)] * _hold_frames(15)

    def update(frame: tuple) -> tuple:
        phase, k = frame
        for idx, line in enumerate(edge_lines):
            if idx < k:
                (x1, y1), (x2, y2) = segments[idx]
                line.set_data([x1, x2], [y1, y2])
        if phase == "path" and path:
            # 木が伸びきったあとで，復元した経路を引く
            p = np.array(path)
            path_plot.set_data(p[:, 0], p[:, 1])
        return edge_lines + [path_plot]

    anim = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
    anim.save(file_name, writer="pillow", fps=15)
    plt.close(fig)


# --- ここから第8章（RRT-Connect）用：2本の木（スタート木・ゴール木）の描画 ---

# スタート木は青，ゴール木は紫で色分けし，どちらから伸びた枝か一目で分かるようにする
TREE_START_COLOR = "tab:blue"     # スタート木の色
TREE_GOAL_COLOR  = "tab:purple"   # ゴール木の色


def _draw_tree_colored(ax: Axes, tree, color: str, label: str,
                       edge_alpha: float = 0.5) -> None:
    """
    探索木を指定色で地図の上に重ねて描く（2本の木を色分けするための内部用）

    パラメータ
        ax: 描画先の Axes
        tree: 探索木（nodes / edge_list() を持つ）
        color: 枝・ノードの色
        label: 凡例ラベル（"start tree" / "goal tree" など）
        edge_alpha: 枝（線）の不透明度
    """
    # 枝（親→子の線分）を先に描いてから，ノードを上に重ねる
    for (x1, y1), (x2, y2) in tree.edge_list():
        ax.plot([x1, x2], [y1, y2], "-", color=color,
                linewidth=0.6, alpha=edge_alpha, zorder=1)

    nodes = tree.nodes
    ax.plot(nodes[:, 0], nodes[:, 1], "o", color=color,
            markersize=2, zorder=2, label=label)


def plot_rrt_connect_trees(grid_map: GridMap, tree_start, tree_goal,
                           file_name: str, title: str = "") -> None:
    """
    スタート木とゴールの2本の木（ノード＋枝）を色分けして地図に重ね，保存する

    パラメータ
        grid_map: 占有格子地図
        tree_start: スタートを根とする木
        tree_goal: ゴールを根とする木
        file_name: 保存ファイル名
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_base(ax, grid_map)
    _draw_tree_colored(ax, tree_start, TREE_START_COLOR, "start tree")
    _draw_tree_colored(ax, tree_goal,  TREE_GOAL_COLOR,  "goal tree")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(file_name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_rrt_connect_path(grid_map: GridMap, tree_start, tree_goal, path: list,
                          file_name: str, title: str = "") -> None:
    """
    2本の木（薄め）の上に，結合して復元した経路を重ねて描いて保存する

    パラメータ
        grid_map: 占有格子地図
        tree_start: スタートを根とする木
        tree_goal: ゴールを根とする木
        path: 経路（連続座標 (x, y) のリスト）
        file_name: 保存ファイル名
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_base(ax, grid_map)
    # 背景に2本の木を薄く描く
    _draw_tree_colored(ax, tree_start, TREE_START_COLOR, "start tree", edge_alpha=0.25)
    _draw_tree_colored(ax, tree_goal,  TREE_GOAL_COLOR,  "goal tree",  edge_alpha=0.25)

    # 結合した経路（橙色の太い折れ線）
    if path:
        p = np.array(path)
        ax.plot(p[:, 0], p[:, 1], "-", color="tab:orange",
                linewidth=2.5, zorder=3, label="path")

    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(file_name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def animate_rrt_connect(grid_map: GridMap, events: list, path: list,
                        file_name: str, step: int = 6, title: str = "") -> None:
    """
    2本の木が交互に枝を伸ばし，中央で出会って経路を引く様子を gif で保存する

    パラメータ
        grid_map: 占有格子地図
        events: 成長の様子（(label, (親座標, 子座標)) を追加順に並べたリスト）
        path: 経路（連続座標 (x, y) のリスト）
        file_name: 保存ファイル名
        step: 1フレームで増やす枝の本数（大きいほど短い gif になる）
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_base(ax, grid_map)
    if title:
        ax.set_title(title)

    # 各枝を「空」の線として用意する。色は生えた木（start/goal）で決める
    edge_lines = []
    for label, _segment in events:
        color = TREE_START_COLOR if label == "start" else TREE_GOAL_COLOR
        edge_lines.append(ax.plot([], [], "-", color=color, linewidth=0.6,
                                  alpha=0.6, zorder=1)[0])
    path_plot, = ax.plot([], [], "-", color="tab:orange", linewidth=2.5, zorder=3)

    n = len(events)
    # 0, step, 2*step, ... と枝を増やし，最後に経路を見せるフレームを足す
    grow_frames = list(range(0, n + 1, step))
    # 最後のコマをくり返して，引けた経路を約1.5秒だけ見せてから先頭へ戻す（gif はループ再生のため）
    frames = [("grow", k) for k in grow_frames] + [("path", n)] * _hold_frames(15)

    def update(frame: tuple) -> tuple:
        phase, k = frame
        for idx, line in enumerate(edge_lines):
            if idx < k:
                _label, (p_parent, p_child) = events[idx]
                line.set_data([p_parent[0], p_child[0]], [p_parent[1], p_child[1]])
        if phase == "path" and path:
            # 2本が伸びきって出会ったあとで，結合した経路を引く
            p = np.array(path)
            path_plot.set_data(p[:, 0], p[:, 1])
        return edge_lines + [path_plot]

    anim = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
    anim.save(file_name, writer="pillow", fps=15)
    plt.close(fig)


# --- ここから第9章（RRT*）用：経路コストの低下グラフと，経路が短くなるアニメ ---

def plot_cost_curve(cost_history: list, file_name: str, title: str = "") -> None:
    """
    反復回数に対する経路コストの低下（最適化が進む様子）を折れ線で描いて保存する

    RRT* はゴール到達後も反復するほど経路コストが下がる。横軸＝試行回数，
    縦軸＝その時点の最良経路コストでプロットし，単調に下がる様子を見せる。

    パラメータ
        cost_history: (試行回数, 最良コスト) のリスト
        file_name: 保存ファイル名
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    if cost_history:
        iters = [c[0] for c in cost_history]
        costs = [c[1] for c in cost_history]
        # コストは「下がったら据え置き」の階段状に変化するので step で描く
        ax.step(iters, costs, where="post", color="tab:blue", linewidth=1.8)
        # 初期解と最終コストに印を付けて，どれだけ短くなったかを分かりやすくする
        ax.plot(iters[0], costs[0], "o", color="tab:red", label=f"initial = {costs[0]:.2f}")
        ax.plot(iters[-1], costs[-1], "o", color="tab:green", label=f"final = {costs[-1]:.2f}")

    ax.set_xlabel("iteration")
    ax.set_ylabel("best path cost")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(file_name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def animate_rrt_star(grid_map: GridMap, tree, snapshots: list,
                     file_name: str, title: str = "") -> None:
    """
    最終的な木（薄め）を背景に，最良経路が反復のたびに短くなっていく様子を gif で保存する

    パラメータ
        grid_map: 占有格子地図
        tree: 成長させた RRT* 探索木（背景に薄く描く）
        snapshots: (試行回数, コスト, 経路) のリスト（最良経路が更新されるたびの記録）
        file_name: 保存ファイル名
        title: 図のタイトル
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_base(ax, grid_map)
    # 背景に最終的な木を薄く描く（経路の改善が際立つように）
    _draw_tree(ax, tree, edge_alpha=0.2)

    path_plot, = ax.plot([], [], "-", color="tab:orange", linewidth=2.5, zorder=3)
    text = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=10,
                   color="tab:orange", verticalalignment="bottom")

    # 各スナップショットを順に見せ，最後の最良経路を少し長く映す
    # このアニメは fps=3（1コマ 約0.33秒）なので，6コマで約2秒。すでに _GOAL_HOLD_SEC を満たす
    frames = list(range(len(snapshots))) + [len(snapshots) - 1] * 6

    def update(k: int) -> tuple:
        iteration, cost, path = snapshots[k]
        p = np.array(path)
        path_plot.set_data(p[:, 0], p[:, 1])
        text.set_text(f"iter={iteration}  cost={cost:.2f}")
        return path_plot, text

    anim = FuncAnimation(fig, update, frames=frames, interval=400, blit=False)
    anim.save(file_name, writer="pillow", fps=3)
    plt.close(fig)
