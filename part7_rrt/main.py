# メイン処理（第7章：RRT でスタートから木を伸ばし，ゴールへ届く経路を引く）

# ライブラリの読み込み
import numpy as np   # 経路長（総延長）の計算に使う

# 自作モジュールの読み込み
from grid_map import make_sample_map     # 共通サンプル地図
from rrt import plan_rrt                 # RRT 経路生成
from plot import plot_rrt_tree, plot_rrt_path, animate_rrt   # 可視化


def _path_length(path: list) -> float:
    """
    経路（連続座標のリスト）の総延長を求める（経路の長さで質を比べるため）

    パラメータ
        path: 経路（連続座標 (x, y) のリスト）

    戻り値
        length: 経路の総延長（空なら 0.0）
    """
    if not path:
        return 0.0
    p = np.array(path)
    return float(np.linalg.norm(p[1:] - p[:-1], axis=1).sum())


def main() -> None:
    """
    メイン処理
    """
    # 共通サンプル地図（壁・部屋・狭い通路を含む 20×20）で RRT を実行
    grid_map = make_sample_map()

    # RRT で経路生成（サンプリング → 最近傍 → 伸ばす → 衝突チェック を繰り返す）
    path, tree, n_iterations = plan_rrt(grid_map)

    print("=== RRT（ランダムに木を伸ばす経路生成）===")
    print(f"反復回数 = {n_iterations} / 木のノード数 = {tree.n_nodes}")
    if path:
        # 経路長（総延長）も出しておく。PRM（第6章）や RRT*（第9章）と「経路の質」を比べるため
        print(f"到達 = True / 経路の通過点数 = {len(path)} / 経路長 = {_path_length(path):.2f}")
    else:
        print("到達 = False（最大反復回数まで木がゴールへ届かなかった）")

    # 図（理論編）：空間を埋めるように広がった探索木＋ゴールへ届いた復元経路（太線）
    plot_rrt_tree(grid_map, tree, "rrt_tree_grow.png", path=path,
                  title="RRT tree (grown to reach goal)")

    # 図（実装編）：探索木（薄め）の上に復元された経路
    plot_rrt_path(grid_map, tree, path, "rrt_path.png",
                  title="RRT path (tree + reconstructed path)")

    # gif（実装編）：枝を数本ずつ伸ばし → ゴールへ届いて経路を引くアニメ（step=3）
    animate_rrt(grid_map, tree, path, "rrt_anime.gif", title="RRT")


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
