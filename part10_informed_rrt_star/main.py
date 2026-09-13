# メイン処理（第10章：Informed RRT* で初期解の後はサンプリングを楕円体に絞り，収束を加速する）

# ライブラリの読み込み
import numpy as np   # 数値計算

# 自作モジュールの読み込み
from grid_map import make_sample_map                       # 共通サンプル地図
from rrt_star import plan_rrt_star                         # 第9章 RRT*（収束の比較相手）
from informed_rrt_star import (plan_informed_rrt_star,
                               rotation_to_world_frame)    # Informed RRT* 経路生成
from plot import (plot_informed_path, animate_informed,
                  plot_cost_compare, plot_ellipse_concept)  # 可視化


def main() -> None:
    """
    メイン処理
    """
    # 共通サンプル地図（壁・部屋・狭い通路を含む 20×20）で Informed RRT* を実行
    grid_map = make_sample_map()

    # Informed RRT* で経路生成（初期解までは一様サンプル，以降は楕円体内サンプルへ切り替わる）
    path, tree, cost_history, snapshots, informed_samples = plan_informed_rrt_star(grid_map)

    print("=== Informed RRT*（初期解後は楕円体サンプリングで収束を加速する経路生成）===")
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

    # 最終的な楕円（最後のスナップショットに記録した c_best に対応する形）
    final_ellipse = snapshots[-1][3] if snapshots else None

    # 図（実装編）：木（薄め）＋楕円体内サンプル＋最終楕円＋最適経路
    plot_informed_path(grid_map, tree, path, final_ellipse, informed_samples,
                       "informed_rrt_star_path.png",
                       title="Informed RRT* path (final ellipse)")

    # gif（実装編）：最良経路が短くなるたびに楕円が細く縮んでいく様子
    animate_informed(grid_map, tree, snapshots, "informed_rrt_star_anime.gif",
                     title="Informed RRT*")

    # --- 第9章 RRT* との収束カーブ比較（同じ地図・同じシードで，初期解後の下がり方を見る）---
    print("\n=== RRT* と Informed RRT* の収束比較（同じ地図・同じシード）===")
    _path9, _tree9, cost_history9, _snap9 = plan_rrt_star(grid_map)
    rrt_star_final = cost_history9[-1][1] if cost_history9 else float("nan")
    informed_final = cost_history[-1][1] if cost_history else float("nan")
    print(f"{'手法':<16}{'最終コスト':>12}")
    print(f"{'RRT*':<18}{rrt_star_final:>10.2f}")
    print(f"{'Informed RRT*':<18}{informed_final:>10.2f}")

    # 図（実装編）：RRT* と Informed RRT* の収束カーブを重ねて比較
    plot_cost_compare(
        [("RRT*", "tab:gray", cost_history9),
         ("Informed RRT*", "tab:blue", cost_history)],
        "informed_vs_rrt_star_cost.png",
        title="RRT* vs Informed RRT* (cost convergence)")

    # --- 図（理論編）：楕円体サンプリングの考え方を見せる概念図 ---
    start_pos = grid_map.cell_to_pos(grid_map.start)
    goal_pos  = grid_map.cell_to_pos(grid_map.goal)
    c_min  = float(np.linalg.norm(goal_pos - start_pos))   # 焦点間の距離
    center = (start_pos + goal_pos) / 2.0
    C      = rotation_to_world_frame(start_pos, goal_pos)

    # 図（理論編）：1つの楕円＋2焦点（スタート・ゴール）で「c_best 以下の領域だけ探す」を示す
    c_best_demo = last_cost if path else c_min * 1.3
    plot_ellipse_concept(
        grid_map,
        [(f"ellipse (c_best={c_best_demo:.1f})", "tab:green",
          (center, C, c_best_demo, c_min))],
        "informed_ellipse.png",
        title=f"Informed sampling ellipse (foci=start/goal, c_min={c_min:.1f})")

    # 図（理論編）：c_best を段階的に小さくした入れ子の楕円（縮むほど細くなる）
    c_levels = [c_min * 1.6, c_min * 1.35, c_min * 1.15, c_min * 1.03]
    colors   = ["tab:purple", "tab:blue", "tab:cyan", "tab:green"]
    ellipses = [(f"c_best={c:.1f}", col, (center, C, c, c_min))
                for c, col in zip(c_levels, colors)]
    plot_ellipse_concept(grid_map, ellipses, "informed_sampling_shrink.png",
                         title="Shrinking ellipse as c_best decreases")


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
