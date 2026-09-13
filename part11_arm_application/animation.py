# 2軸ロボットアームの動作アニメーション（gif）の作成
#
# 経路生成手法が返すのは「関節空間 (θ1, θ2) 上の中継点の列」です。それを順運動学で
# アームの姿勢に直し，障害物をよけながら動く様子を gif にします。前作 src/two_dof/part4_rrt_fcl の
# RobotAnimation を，本章の C-space地図（座標 (x, y) ⇔ 関節角度 (θ1, θ2)）に合わせて作り直したものです。

# ライブラリの読み込み
import numpy as np                      # 数値計算
import matplotlib.pyplot as plt         # 描画用
import matplotlib.animation as ani      # アニメーション用
import matplotlib.patches as patches    # 2次元形状の描画

# 自作モジュールの読み込み
from constant import *                  # 定数
from grid_map import CSpaceMap          # 関節空間（C-space）地図


# アームの可動範囲（リンク長の合計）に少し余白を足した描画範囲 [m]
_AXIS_LIMIT = sum(ARM_LINK_LENGTHS) + 0.3
# 関節空間で中継点のあいだを補間するときの刻み幅 [rad]（細かいほどなめらかだがフレームが増える）
_INTERPOLATION_STEP = 0.15
# 1コマの表示時間 [ms]（ArtistAnimation の interval。gif の速さを決める）
_FRAME_INTERVAL_MS = 60
# ゴールに着いたあと，最後のコマを見せ続ける時間 [s]。gif はくり返し再生されるので，
# 到達した姿勢をしばらく見せてから先頭へ戻さないと，見る側には速すぎて分かりにくい
_GOAL_HOLD_SEC = 1.5
# 上を「くり返すコマ数」に直したもの（1.5秒 ÷ 60ms = 25コマ）
_GOAL_HOLD_FRAMES = max(round(_GOAL_HOLD_SEC * 1000 / _FRAME_INTERVAL_MS), 1)


def _interpolate_joints(joints: list) -> list:
    """
    関節空間の中継点列のあいだを細かく補間して，なめらかなアニメ用の姿勢列にする（内部用）

    経路の中継点は飛び飛び（グラフ系は1セル＝5°，サンプリング系は1ステップ＝十数°）なので，
    そのまま動かすとカクつく。隣り合う2姿勢のあいだを一定刻みで埋める。

    パラメータ
        joints: 関節角度 (θ1, θ2) の中継点リスト

    戻り値
        dense: 補間後の関節角度の列
    """
    dense = []
    for current, nxt in zip(joints[:-1], joints[1:]):
        difference = nxt - current
        # 2姿勢間の関節の動き量に応じて分割数を決める（最低1分割）
        n_divided = max(int(np.linalg.norm(difference) / _INTERPOLATION_STEP), 1)
        for k in range(n_divided):
            dense.append(current + k / n_divided * difference)
    dense.append(joints[-1])
    return dense


class ArmAnimation:
    """
    2軸ロボットアームの動作アニメーション作成

    プロパティ
        _figure: 描画枠
        _axis: 描画内容

    メソッド
        public
            plot_animation(): 関節空間の経路からアーム動作の gif を作る
    """

    def _reset(self, title: str) -> None:
        """
        描画枠を初期化する

        パラメータ
            title: グラフのタイトル
        """
        self._figure = plt.Figure()
        self._axis = self._figure.add_subplot(111)
        self._axis.set_xlabel("X [m]")
        self._axis.set_ylabel("Y [m]")
        self._axis.set_title(title)
        self._axis.set_xlim(-_AXIS_LIMIT, _AXIS_LIMIT)
        self._axis.set_ylim(-_AXIS_LIMIT, _AXIS_LIMIT)
        self._axis.set_aspect("equal")
        self._axis.grid()

    def _plot_environment(self, environment) -> None:
        """
        障害物（円・長方形）を描画する

        パラメータ
            environment: 障害物のある環境（Robot2DEnv）
        """
        for name, datas in environment.interferences.items():
            if name == INTERFERENCE.CIRCLE:
                for x, y, radius in datas:
                    self._axis.add_patch(patches.Circle((x, y), radius, color="gray", alpha=0.5))
            elif name == INTERFERENCE.RECTANGLE:
                for x, y, center_x, center_y, angle in datas:
                    # 中心点から左下隅へずらして長方形を置く
                    self._axis.add_patch(patches.Rectangle((center_x - x / 2, center_y - y / 2),
                                                           x, y, angle=angle, color="gray", alpha=0.5))

    def plot_animation(self, cspace_map: CSpaceMap, path_coords: list,
                       anime_file_name: str, title: str = "") -> None:
        """
        関節空間の経路（座標列）から，アームが障害物をよけて動く gif を作る

        パラメータ
            cspace_map: 関節空間（C-space）地図（アーム・環境を保持）
            path_coords: 経路（連続座標 (x, y) のリスト。空なら何もしない）
            anime_file_name: 出力する gif のファイル名
            title: グラフのタイトル
        """
        if len(path_coords) == 0:
            # 経路が無い（到達失敗）ときは gif を作らない
            print(f"  （経路が無いため {anime_file_name} は作成しません）")
            return

        robot       = cspace_map.robot
        environment = cspace_map.environment

        # 経路の座標を関節角度へ直し，なめらかになるよう補間する
        joints = [cspace_map.coord_to_joint(np.asarray(point, dtype=float)) for point in path_coords]
        dense_joints = _interpolate_joints(joints)

        self._reset(title)
        self._plot_environment(environment)

        # スタート・ゴールの手先位置を点で示す（水色＝スタート，赤＝最終目標）
        # 「経路の端」ではなく地図が持つスタート・ゴールセルから求める。こうしておくと，
        # ポテンシャル法のように途中で止まる（未到達の）手法でも，目指していた最終目標が赤点で分かる
        start_joint = cspace_map.coord_to_joint(cspace_map.cell_to_pos(cspace_map.start))
        goal_joint  = cspace_map.coord_to_joint(cspace_map.cell_to_pos(cspace_map.goal))
        start_pos = robot.forward_kinematics(start_joint)
        goal_pos  = robot.forward_kinematics(goal_joint)
        start_image = self._axis.scatter(start_pos[0], start_pos[1], color="cyan", zorder=5)
        goal_image  = self._axis.scatter(goal_pos[0],  goal_pos[1],  color="red",  zorder=5)

        imgs = []
        # 手先位置の軌跡を貯めながら1コマずつ描く
        trajectory = np.zeros((len(dense_joints), DIMENSION_2D))
        for i, joint in enumerate(dense_joints):
            frame = []
            # 順運動学で全リンク（根もと→関節→手先）の位置を求めて線で結ぶ
            all_link_pos = robot.forward_kinematics_all_link_pos(joint)
            frame.extend(self._axis.plot(all_link_pos[:, 0], all_link_pos[:, 1], color="blue"))
            frame.append(self._axis.scatter(all_link_pos[:, 0], all_link_pos[:, 1], color="black", alpha=0.5))
            # 手先の軌跡を緑線で残す
            trajectory[i] = all_link_pos[-1]
            frame.extend(self._axis.plot(trajectory[:i + 1, 0], trajectory[:i + 1, 1], color="lime"))
            frame.extend([start_image, goal_image])
            imgs.append(frame)

        # 最後のコマ（ゴールに着いた姿勢）をくり返し足して，先頭へ戻る前に少し静止させる
        imgs.extend([imgs[-1]] * _GOAL_HOLD_FRAMES)

        # アニメーションを作成して保存（gif 出力には pillow を使う）
        animation = ani.ArtistAnimation(self._figure, imgs, interval=_FRAME_INTERVAL_MS)
        animation.save(anime_file_name, writer="pillow")
        plt.close(self._figure)
