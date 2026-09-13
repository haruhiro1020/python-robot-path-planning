# メイン処理（第6章：PRM で自由空間にロードマップを作り，A* で最短経路を引く）

# 自作モジュールの読み込み
from grid_map import make_sample_map     # 共通サンプル地図
from prm import plan_prm                 # PRM 経路生成
from plot import plot_roadmap, plot_prm_path, animate_prm   # 可視化


def main() -> None:
    """
    メイン処理
    """
    # 共通サンプル地図（壁・部屋・狭い通路を含む 20×20）で PRM を実行
    grid_map = make_sample_map()

    # PRM で経路生成（構築フェーズ：点を撒いて辺でつなぐ → クエリフェーズ：A* で最短を引く）
    path, roadmap, cost = plan_prm(grid_map)

    print("=== PRM（確率的ロードマップ法）===")
    print(f"サンプル数 N = {len(roadmap.nodes) - 2}（＋スタート・ゴールの2点）")
    print(f"ロードマップ：ノード数 = {roadmap.n_nodes} / 辺の本数 = {roadmap.n_edges}")
    if path:
        print(f"到達 = True / 経由ノード数 = {len(path)} / 経路コスト = {cost:.2f}")
    else:
        print("到達 = False（ロードマップ上でスタートとゴールがつながらなかった）")

    # 図（理論編）：撒いた点＋接続された辺＝ロードマップそのもの
    plot_roadmap(grid_map, roadmap, "prm_roadmap.png",
                 title="PRM roadmap (sampled points + edges)")

    # 図（実装編）：ロードマップ（薄め）の上に引かれた経路
    plot_prm_path(grid_map, roadmap, path, "prm_path.png",
                  title="PRM path (roadmap + shortest path)")

    # gif（実装編）：点を撒く → 辺でつなぐ → 経路を引く の3フェーズ
    n_samples = len(roadmap.nodes) - 2   # スタート・ゴールを除いたサンプル点の数
    animate_prm(grid_map, roadmap, path, n_samples, "prm_anime.gif", title="PRM")


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
