# ロボットアームの運動学を記載

# ライブラリの読み込み
from abc import ABC, abstractmethod   # 抽象クラス（それ自体は使わず、継承のひな型にするクラス）を作る標準ライブラリ
import numpy as np

# サードパーティーの読み込み
import fcl

# 自作モジュールの読み込み
from constant import *              # 定数
from rotation import Rotation       # 回転行列


class Robot(ABC):
    """
    ロボットのベースクラス(抽象クラス)

    プロパティ
        _links(numpy.ndarray): ロボットのリンク長 [m]
        _rot(Rotation): 回転行列クラス
        _objects(list): 干渉物オブジェクト
        _manager(fcl.DynamicAABBTreeCollisionManager): 干渉判定クラス

    メソッド
        public
            forward_kinematics(): 順運動学 (ロボットの関節角度からロボットの手先位置を算出)
            inverse_kinematics(): 逆運動学 (ロボットの手先位置からロボットの関節角度を算出)
            forward_kinematics_all_link_pos(): 順運動学で全リンクの位置を取得
            update(): 角度を与えて，各リンクの直方体を更新する
            links(): _linksプロパティのゲッター
            manager(): _managerプロパティのゲッター
    """
    # 定数の定義
    _DIMENSION_POSE  = DIMENSION_NONE       # 手先位置の次元数
    _DIMENSION_THETA = DIMENSION_NONE       # 関節角度の次元数
    _DIMENSION_LINK  = DIMENSION_NONE       # リンク数
    _DIMENSION_AXIS  = DIMENSION_NONE       # 回転軸数

    _INITIAL_THETA   = 0.0                  # 初期回転角度 [rad]


    def __init__(self, links: np.ndarray) -> None:
        """
        コンストラクタ

        パラメータ
            links: ロボットのリンク長 [m]
        """
        if np.size(links) != self._DIMENSION_LINK:
            # 異常
            raise ValueError(f"links's size is abnormal. correct is {self._DIMENSION_LINK}")

        # プロパティの初期化
        self._links = links
        self._rot = Rotation()
        self._objects = []
        self._manager = None

    @property
    def links(self) -> np.ndarray:
        """
        _linksプロパティのゲッター
        """
        return self._links

    @property
    def manager(self) -> fcl.DynamicAABBTreeCollisionManager:
        """
        _managerプロパティのゲッター
        """
        return self._manager

    @abstractmethod
    def forward_kinematics(self, thetas: np.ndarray) -> np.ndarray:
        """
        順運動学 (ロボットの関節角度からロボットの手先位置を算出)

        パラメータ
            thetas: ロボットの関節角度 [rad]

        戻り値
            pose: ロボットの手先位置 (位置 + 姿勢) [m] + [rad]
        """
        # 抽象メソッド：中身は書かず、子クラス側で必ずオーバーライドする

    @abstractmethod
    def inverse_kinematics(self, pose: np.ndarray) -> np.ndarray:
        """
        逆運動学 (ロボットの手先位置からロボットの関節角度を算出)

        パラメータ
            pose: ロボットの手先位置 (位置 + 姿勢) [m] + [rad]

        戻り値
            thetas: ロボットの関節角度 [rad]
        """
        # 抽象メソッド：中身は書かず、子クラス側で必ずオーバーライドする

    @abstractmethod
    def forward_kinematics_all_link_pos(self, thetas: np.ndarray) -> np.ndarray:
        """
        順運動学で全リンクの位置を取得

        パラメータ
            thetas: ロボットの関節角度 [rad]

        戻り値
            all_link_pose: ロボットの全リンク位置 (位置 + 姿勢) [m] + [rad]
        """
        # 抽象メソッド：中身は書かず、子クラス側で必ずオーバーライドする

    @abstractmethod
    def update(self, thetas: np.ndarray) -> None:
        """
        角度を与えて，各リンクの直方体を更新する

        パラメータ
            thetas: ロボットの関節角度 [rad]
        """
        # 抽象メソッド：中身は書かず、子クラス側で必ずオーバーライドする


class Robot2DoF(Robot):
    """
    2軸ロボットクラス

    プロパティ
        _links(numpy.ndarray): ロボットのリンク長
        _rot(Rotation): 回転行列クラス
        _axiss(list): 関節の回転軸

    メソッド
        public
            forward_kinematics(): 順運動学 (ロボットの関節角度からロボットの手先位置を算出)
            inverse_kinematics(): 逆運動学 (ロボットの手先位置からロボットの関節角度を算出)
            forward_kinematics_all_link_pos(): 順運動学で全リンクの位置を取得
            update(): 角度を与えて，各リンクの直方体を更新する
        protected
            _link_transforms(): 各リンク(直方体)の位置・姿勢を計算する
    """
    # 定数の定義
    _DIMENSION_POSE  = DIMENSION_2D         # 手先位置の次元数
    _DIMENSION_THETA = DIMENSION_2D         # 関節角度の次元数
    _DIMENSION_LINK  = DIMENSION_2D         # リンク数

    _DETERMINANT_THRESHOLD = 1e-4           # 行列式の閾値
    _BOX_WIDTH = 1e-2                       # 各リンクの幅を定義


    def __init__(self, links: np.ndarray) -> None:
        """
        コンストラクタ

        パラメータ
            links: ロボットのリンク長 [m]
        """
        # 親クラスの初期化
        super().__init__(links)

        # リンク1とリンク2は回転軸がz軸である
        self._axiss = [ROTATION_Z_AXIS, ROTATION_Z_AXIS]
        # 初期角度
        initial_thetas = np.zeros(self._DIMENSION_THETA)

        # ロボットの各リンクを直方体として定義する
        self._objects = []
        # 初期角度での各リンク(直方体)の位置・姿勢を計算
        transforms = self._link_transforms(initial_thetas)
        for i, transform in enumerate(transforms):
            # 直方体の定義 (x, y, zの長さ。y, z方向はごく細くする)
            box = fcl.Box(self._links[i], self._BOX_WIDTH * 2, self._BOX_WIDTH * 2)
            # 直方体に位置・姿勢を与えてモデルを追加
            self._objects.append(fcl.CollisionObject(box, transform))

        # 直方体をAABBとして，DynamicAABBTreeCollisionManager に登録
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.registerObjects(self._objects)
        self._manager.setup()

    def forward_kinematics(self, thetas: np.ndarray) -> np.ndarray:
        """
        順運動学 (ロボットの関節角度からロボットの手先位置を算出)

        パラメータ
            thetas: ロボットの関節角度 [rad]

        戻り値
            pose: ロボットの手先位置 (位置) [m]
        """
        # パラメータの次元数を確認
        if np.size(thetas) != self._DIMENSION_THETA:
            raise ValueError(f"thetas's size is abnormal. thetas's size is {np.size(thetas)}")

        # あらかじめ三角関数を算出する
        sin1  = np.sin(thetas[0])
        sin12 = np.sin(thetas[0] + thetas[1])
        cos1  = np.cos(thetas[0])
        cos12 = np.cos(thetas[0] + thetas[1])

        theta1  = np.array([cos1,  sin1])
        theta12 = np.array([cos12, sin12])

        # ロボットの手先位置を算出
        pose = self._links[0] * theta1 + self._links[1] * theta12

        return pose

    def inverse_kinematics(self, pose: np.ndarray, upper: bool = False) -> np.ndarray:
        """
        逆運動学 (ロボットの手先位置からロボットの関節角度を算出)

        パラメータ
            pose: ロボットの手先位置 (位置) [m]
            upper: 腕が上向かどうか

        戻り値
            thetas: ロボットの関節角度 [rad]
        """
        # パラメータの次元数を確認
        if np.size(pose) != self._DIMENSION_POSE:
            raise ValueError(f"parameter pose's size is abnormal. pose's size is {np.size(pose)}")

        # c2 = {(px ** 2 + py ** 2) - (l1 ** 2 + l2 ** 2)} / (2 * l1 * l2)
        px = pose[0]
        py = pose[1]
        l1 = self._links[0]
        l2 = self._links[1]
        cos2  = ((px ** 2 + py ** 2) - (l1 ** 2 + l2 ** 2)) / (2 * l1 * l2)
        # cosの範囲は-1以上1以下である
        if cos2 < -1 or cos2 > 1:
            # 異常
            raise ValueError(f"cos2 is abnormal. cos2 is {cos2}")

        # sinも求めて，theta2をatan2()より算出する
        sin2 = np.sqrt(1 - cos2 ** 2)
        theta2 = np.arctan2(sin2,  cos2)
        if not upper:
            # 下向きの角度のため，三角関数も更新
            theta2 = -theta2
            sin2 = np.sin(theta2)
            cos2 = np.cos(theta2)

        # 行列計算
        # [c1, s1] = [[l1 + l2 * c2, -l2 * s2], [l2 * s2, l1 + l2 * c2]] ** -1 * [px, py]
        element1 =  l1 + l2 * cos2
        element2 = -l2 * sin2
        matrix = np.array([[ element1, element2],
                           [-element2, element1]])
        # 行列式を計算
        det = np.linalg.det(matrix)
        # 0近傍の確認
        if abs(det) <= self._DETERMINANT_THRESHOLD:
            # 0近傍 (異常)
            raise ValueError(f"det is abnormal. det is {det}")

        # [c1, s1]の計算
        cos1_sin1 = np.dot(np.linalg.inv(matrix), pose)
        # theta1をatan2()より算出する
        theta1 = np.arctan2(cos1_sin1[1], cos1_sin1[0])

        thetas = np.array([theta1, theta2])

        return thetas

    def forward_kinematics_all_link_pos(self, thetas: np.ndarray) -> np.ndarray:
        """
        順運動学で全リンクの位置を取得 (グラフの描画で使用する)

        パラメータ
            thetas: ロボットの関節角度 [rad]

        戻り値
            all_link_pose: ロボットの全リンク位置 (位置 + 姿勢) [m] + [rad]
        """
        # パラメータの次元数を確認
        if np.size(thetas) != self._DIMENSION_THETA:
            raise ValueError(f"thetas's size is abnormal. thetas's size is {np.size(thetas)}")

        # 回転角度とリンク長をローカル変数に保存
        theta1 = thetas[0]
        theta2 = thetas[1]
        link1  = self._links[0]
        link2  = self._links[1]

        # 三角関数を計算
        s1  = np.sin(theta1)
        c1  = np.cos(theta1)
        s12 = np.sin(theta1 + theta2)
        c12 = np.cos(theta1 + theta2)

        # 各リンクの位置を算出
        base_pos  = np.zeros(self._DIMENSION_POSE)
        link1_pos = link1 * np.array([c1, s1])
        link2_pos = link1_pos + link2 * np.array([c12, s12])

        # 全リンクの位置を算出
        all_link_pose = np.array([base_pos, link1_pos, link2_pos])

        return all_link_pose

    def _link_transforms(self, thetas: np.ndarray) -> list:
        """
        各リンク(直方体)の位置・姿勢(fcl.Transform)を計算する

        パラメータ
            thetas: ロボットの関節角度 [rad]

        戻り値
            transforms: 各リンクの位置・姿勢(fcl.Transform)のリスト
        """
        # 順運動学により，全リンク(ベースリンク，リンク1，リンク2)の位置を計算
        all_link_pose = self.forward_kinematics_all_link_pos(thetas)

        transforms = []
        # 1つ前のリンクの回転行列(最初は単位行列)
        prev_rotation = np.eye(3)

        # ロボットの各リンクの位置・姿勢を計算する
        for i in range(self._DIMENSION_THETA):
            # 各リンクの回転行列を定義
            rotation = self._rot.rot(thetas[i], self._axiss[i])
            # 1つ前のリンクの回転も考慮する
            rotation = np.dot(prev_rotation, rotation)
            # 各リンクの中心位置 (x, y, z) を定義
            center = np.zeros(DIMENSION_3D)
            center[:self._DIMENSION_POSE] = (all_link_pose[i + 1] - all_link_pose[i]) / 2 + all_link_pose[i]
            # 直方体の位置・姿勢を保存
            transforms.append(fcl.Transform(rotation, center))
            # 1つ前のリンクの回転行列を更新
            prev_rotation = rotation

        return transforms

    def update(self, thetas: np.ndarray) -> None:
        """
        角度を与えて，各リンクの直方体を更新する

        パラメータ
            thetas: ロボットの関節角度 [rad]
        """
        # パラメータの次元数を確認
        if np.size(thetas) != self._DIMENSION_THETA:
            raise ValueError(f"thetas's size is abnormal. thetas's size is {np.size(thetas)}")

        # この角度での各リンク(直方体)の位置・姿勢を計算
        transforms = self._link_transforms(thetas)
        # 各直方体の位置・姿勢を更新
        for obj, transform in zip(self._objects, transforms):
            obj.setTransform(transform)

        # AABBを更新
        self._manager.update()
