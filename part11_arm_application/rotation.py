# 回転行列の定義

# 標準ライブラリの読み込み
import numpy as np


# サードパーティーの読み込み


# 自作モジュールの読み込み
from constant import *      # 定数



class Rotation:
    """
    回転行列クラス
    """

    def _rot_x(self, theta: float) -> np.ndarray:
        """
        x軸方向にtheta[rad]回転させる回転行列

        パラメータ
            theta: 回転角度 [rad]

        戻り値
            rotation: 回転行列
        """
        rotation = np.array([[1, 0,              0            ],
                             [0, np.cos(theta), -np.sin(theta)],
                             [0, np.sin(theta),  np.cos(theta)]])

        return rotation

    def _rot_y(self, theta: float) -> np.ndarray:
        """
        y軸方向にtheta[rad]回転させる回転行列

        パラメータ
            theta: 回転角度 [rad]

        戻り値
            rotation: 回転行列
        """
        rotation = np.array([[ np.cos(theta), 0, np.sin(theta)],
                             [ 0,             1, 0            ],
                             [-np.sin(theta), 0, np.cos(theta)]])

        return rotation

    def _rot_z(self, theta: float) -> np.ndarray:
        """
        z軸方向にtheta[rad]回転させる回転行列

        パラメータ
            theta: 回転角度 [rad]

        戻り値
            rotation: 回転行列
        """
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                             [np.sin(theta),  np.cos(theta), 0],
                             [0,              0,             1]])

        return rotation

    def rot(self, theta: float, axis: str) -> np.ndarray:
        """
        回転軸に応じた回転行列の取得

        パラメータ
            theta: 回転角度 [rad]
            axis: 回転軸

        戻り値
            rotation: 回転行列
        """
        if axis == ROTATION_X_AXIS:
            # 回転軸がx軸
            rotation = self._rot_x(theta)
        elif axis == ROTATION_Y_AXIS:
            # 回転軸がy軸
            rotation = self._rot_y(theta)
        elif axis == ROTATION_Z_AXIS:
            # 回転軸がz軸
            rotation = self._rot_z(theta)
        else:
            # 異常
            raise ValueError(f"axis is abnormal. now is {axis}")

        return rotation
