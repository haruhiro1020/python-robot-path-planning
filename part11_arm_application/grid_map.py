# 関節空間（C-space）を格子に区切った地図のクラスと，そのサンプル地図づくり
#
# 第2章の GridMap は「点ロボットが動く実空間の2次元グリッド」でした。本章ではそれを
# 「2軸アームの関節空間 (θ1, θ2) を格子に区切った地図（C-space地図）」として作り直します。
# 違いは1点だけ：各セルが障害物かどうかを，
#   点ロボット → そのセルが障害物に重なるか
#   アーム     → その関節角度 (θ1, θ2) のときアームが障害物にぶつかるか（python-fcl で判定）
# で決めることです。一度この地図を作ってしまえば，第3〜10章の探索・サンプリングのコードが
# 座標 (x, y) を関節角度 (θ1, θ2) と読み替えるだけでそのまま動きます。

# ライブラリの読み込み
import numpy as np

# 自作モジュールの読み込み
from constant import *                  # 定数
from robot import Robot2DoF             # 2軸ロボットアーム（順運動学・逆運動学・fcl干渉判定）
from environment import Robot2DEnv      # 障害物のある環境（前作 part4_rrt_fcl と同じ）


class GridMap:
    """
    占有格子地図クラス（2次元グリッドを 0=空き / 1=障害物 で表す地図）

    第2章のものと同一。本章ではこの「2次元グリッド」を関節空間 (θ1, θ2) の地図として使う。
    グラフ探索（ダイクストラ法/A*）はこの離散化した格子の上を進む。

    プロパティ
        _grid(numpy.ndarray): 地図本体（0=空き / 1=障害物 の2次元配列）
        _start(tuple): スタートセル (row, col)
        _goal(tuple): ゴールセル (row, col)

    メソッド
        public
            grid(): _gridプロパティのゲッター
            shape(): 地図の大きさ (行数, 列数) を取得
            start(): _startプロパティのゲッター
            goal(): _goalプロパティのゲッター
            in_bounds(): セルが地図の範囲内かを判定
            is_obstacle(): セルが障害物（または範囲外）かを判定
            is_free(): セルが通行できる空きセルかを判定
            pos_to_cell(): 連続座標(x, y)をセル(row, col)へ変換
            cell_to_pos(): セル(row, col)を連続座標(x, y)へ変換
            neighbors(): あるセルの近傍セルと移動コストを取得
    """

    def __init__(self, grid: np.ndarray, start: tuple = None, goal: tuple = None) -> None:
        """
        コンストラクタ

        パラメータ
            grid: 地図本体（0=空き / 1=障害物 の2次元配列）
            start: スタートセル (row, col)
            goal: ゴールセル (row, col)
        """
        self._grid  = np.array(grid, dtype=int)
        self._start = start
        self._goal  = goal

    @property
    def grid(self) -> np.ndarray:
        """
        _gridプロパティのゲッター
        """
        return self._grid

    @property
    def shape(self) -> tuple:
        """
        地図の大きさ (行数, 列数) を取得
        """
        return self._grid.shape

    @property
    def start(self) -> tuple:
        """
        _startプロパティのゲッター
        """
        return self._start

    @property
    def goal(self) -> tuple:
        """
        _goalプロパティのゲッター
        """
        return self._goal

    def in_bounds(self, cell: tuple) -> bool:
        """
        セルが地図の範囲内かを判定

        パラメータ
            cell: セル (row, col)

        戻り値
            is_inside: True / False = 範囲内 / 範囲外
        """
        row, col = cell
        rows, cols = self._grid.shape
        is_inside = (0 <= row < rows) and (0 <= col < cols)
        return is_inside

    def is_obstacle(self, cell: tuple) -> bool:
        """
        セルが障害物（または範囲外）かを判定

        パラメータ
            cell: セル (row, col)

        戻り値
            is_obstacle: True / False = 障害物（範囲外含む） / 空き
        """
        if not self.in_bounds(cell):
            # 範囲外は障害物として扱う（地図の外には出られない）
            return True
        return self._grid[cell[0], cell[1]] == CELL_OBSTACLE

    def is_free(self, cell: tuple) -> bool:
        """
        セルが通行できる空きセルかを判定

        パラメータ
            cell: セル (row, col)

        戻り値
            is_free: True / False = 空き / 障害物（範囲外含む）
        """
        if not self.in_bounds(cell):
            # 範囲外は通行できない（地図の外には出られない）
            return False
        return self._grid[cell[0], cell[1]] == CELL_FREE

    def pos_to_cell(self, pos: np.ndarray) -> tuple:
        """
        連続座標(x, y)をセル(row, col)へ変換

        パラメータ
            pos: 連続座標 (x, y)

        戻り値
            cell: セル (row, col)
        """
        # x が列(col)，y が行(row) に対応する（四捨五入で最寄りのセルへ）
        x, y = pos
        cell = (int(round(y)), int(round(x)))
        return cell

    def cell_to_pos(self, cell: tuple) -> np.ndarray:
        """
        セル(row, col)を連続座標(x, y)へ変換

        パラメータ
            cell: セル (row, col)

        戻り値
            pos: 連続座標 (x, y)
        """
        row, col = cell
        pos = np.array([col, row], dtype=float)
        return pos

    def neighbors(self, cell: tuple, connectivity: CONNECTIVITY = CONNECTIVITY.EIGHT) -> list:
        """
        あるセルの近傍セルと移動コストを取得

        パラメータ
            cell: 中心となるセル (row, col)
            connectivity: 近傍の種類（4近傍 / 8近傍）

        戻り値
            result: [(近傍セル, 移動コスト), ...] のリスト（障害物・範囲外は除く）
        """
        row, col = cell

        # 4近傍（上下左右）の移動方向とコスト
        moves = [(-1,  0, COST_STRAIGHT), (1, 0, COST_STRAIGHT),
                 ( 0, -1, COST_STRAIGHT), (0, 1, COST_STRAIGHT)]
        if connectivity == CONNECTIVITY.EIGHT:
            # 斜め4方向を追加（コストは √2）
            moves += [(-1, -1, COST_DIAGONAL), (-1, 1, COST_DIAGONAL),
                      ( 1, -1, COST_DIAGONAL), ( 1, 1, COST_DIAGONAL)]

        result = []
        for d_row, d_col, cost in moves:
            n_cell = (row + d_row, col + d_col)
            if self.is_obstacle(n_cell):
                # 障害物・範囲外には進めない
                continue
            if d_row != 0 and d_col != 0:
                # 斜め移動のときは，間の角（縦横セル）が障害物なら通さない（壁の角抜け防止）
                if self.is_obstacle((row, col + d_col)) or self.is_obstacle((row + d_row, col)):
                    continue
            result.append((n_cell, cost))

        return result


class CSpaceMap(GridMap):
    """
    関節空間（C-space）地図クラス（GridMap に「セル ⇔ 関節角度」の対応づけを足したもの）

    格子のセル (row, col) や連続座標 (x, y) を，アームの関節角度 (θ1, θ2) と相互変換できる。
    また，衝突判定で使うアーム本体（robot）と障害物環境（environment）を保持する。
    第3〜10章のアルゴリズムは座標 (x, y) を扱うので，この地図を渡すだけでそのまま関節空間を探索できる。

    プロパティ（GridMap に追加）
        _robot(Robot2DoF): 2軸ロボットアーム（衝突判定で使う）
        _environment(Robot2DEnv): 障害物のある環境（衝突判定で使う）

    メソッド（GridMap に追加）
        public
            robot(): _robotプロパティのゲッター
            environment(): _environmentプロパティのゲッター
            coord_to_joint(): 連続座標 (x, y) を関節角度 (θ1, θ2) へ変換
            joint_to_coord(): 関節角度 (θ1, θ2) を連続座標 (x, y) へ変換
    """

    def __init__(self, grid: np.ndarray, start: tuple, goal: tuple,
                 robot: Robot2DoF, environment: Robot2DEnv) -> None:
        """
        コンストラクタ

        パラメータ
            grid: C-space地図本体（0=空き / 1=禁止領域 の2次元配列）
            start: スタートセル (row, col)
            goal: ゴールセル (row, col)
            robot: 2軸ロボットアーム
            environment: 障害物のある環境
        """
        super().__init__(grid, start, goal)
        self._robot       = robot
        self._environment = environment

    @property
    def robot(self) -> Robot2DoF:
        """
        _robotプロパティのゲッター（collision.py が衝突判定に使う）
        """
        return self._robot

    @property
    def environment(self) -> Robot2DEnv:
        """
        _environmentプロパティのゲッター（collision.py が衝突判定に使う）
        """
        return self._environment

    def coord_to_joint(self, coord: np.ndarray) -> np.ndarray:
        """
        連続座標 (x, y) を関節角度 (θ1, θ2) [rad] へ変換

        x（列方向）を θ1，y（行方向）を θ2 に対応づける。座標 0〜(分割数-1) が
        関節角度 JOINT_MIN〜JOINT_MAX に線形に対応する。

        パラメータ
            coord: 連続座標 (x, y)

        戻り値
            joint: 関節角度 (θ1, θ2) [rad]
        """
        rows, cols = self._grid.shape
        x, y = coord
        theta1 = JOINT_MIN + (x / (cols - 1)) * (JOINT_MAX - JOINT_MIN)
        theta2 = JOINT_MIN + (y / (rows - 1)) * (JOINT_MAX - JOINT_MIN)
        return np.array([theta1, theta2])

    def joint_to_coord(self, joint: np.ndarray) -> np.ndarray:
        """
        関節角度 (θ1, θ2) [rad] を連続座標 (x, y) へ変換（coord_to_joint の逆）

        パラメータ
            joint: 関節角度 (θ1, θ2) [rad]

        戻り値
            coord: 連続座標 (x, y)
        """
        rows, cols = self._grid.shape
        theta1, theta2 = joint
        x = (theta1 - JOINT_MIN) / (JOINT_MAX - JOINT_MIN) * (cols - 1)
        y = (theta2 - JOINT_MIN) / (JOINT_MAX - JOINT_MIN) * (rows - 1)
        return np.array([x, y])


def _arm_collides(robot: Robot2DoF, environment: Robot2DEnv, joint: np.ndarray) -> bool:
    """
    その関節角度のとき，アームが障害物にぶつかるかを python-fcl で判定する（内部用）

    パラメータ
        robot: 2軸ロボットアーム
        environment: 障害物のある環境
        joint: 関節角度 (θ1, θ2) [rad]

    戻り値
        is_collision: True / False = ぶつかる（禁止領域） / ぶつからない
    """
    # アームの各リンク（直方体）をこの角度に動かし，環境と干渉判定する
    robot.update(joint)
    return environment.is_collision_dist(robot.manager, margin=COLLISION_MARGIN)


def _find_free_cell(grid: np.ndarray, cell: tuple) -> tuple:
    """
    指定セルが禁止領域なら，らせん状に最寄りの空きセルを探して返す（内部用）

    逆運動学で求めたスタート・ゴールの関節角度が，格子に丸めるとたまたま禁止領域の
    セルに乗ってしまうことがある。そのときは一番近い空きセルへずらす。

    パラメータ
        grid: C-space地図本体
        cell: 中心セル (row, col)

    戻り値
        free_cell: 最寄りの空きセル (row, col)
    """
    rows, cols = grid.shape
    row0, col0 = cell
    # 距離0（自分自身）から順に，外側のリングへ広げながら空きセルを探す
    for radius in range(max(rows, cols)):
        for d_row in range(-radius, radius + 1):
            for d_col in range(-radius, radius + 1):
                row, col = row0 + d_row, col0 + d_col
                if 0 <= row < rows and 0 <= col < cols and grid[row, col] == CELL_FREE:
                    return (row, col)
    raise ValueError("空きセルが1つも見つかりませんでした（地図が障害物で埋まっています）")


def make_cspace_map(resolution: int = CSPACE_RESOLUTION) -> CSpaceMap:
    """
    本章で共通して使う C-space地図（2軸アームの関節空間を格子に区切った地図）を作成する

    手順は3つ：
      ① 2軸アームと障害物環境を用意する
      ② 関節空間 (θ1, θ2) を resolution×resolution の格子に区切り，各セルでアームが
         障害物にぶつかるかを python-fcl で判定して，禁止領域（C-obstacle）を 1 で塗る
      ③ 始点・終点の手先位置を逆運動学で関節角度へ直し，格子のセルへ対応づける

    パラメータ
        resolution: 関節空間を区切る格子の1辺のセル数（細かいほど精密だが構築が重い）

    戻り値
        cspace_map: C-space地図（スタート・ゴール・アーム・環境を保持）
    """
    # ① アームと環境を用意
    robot       = Robot2DoF(np.array(ARM_LINK_LENGTHS))
    environment = Robot2DEnv()

    # ② 関節空間を格子に区切り，禁止領域を塗る
    #    行(row)が θ2，列(col)が θ1 に対応する（coord_to_joint と同じ並び）
    thetas = np.linspace(JOINT_MIN, JOINT_MAX, resolution)
    grid   = np.zeros((resolution, resolution), dtype=int)
    for col, theta1 in enumerate(thetas):
        for row, theta2 in enumerate(thetas):
            if _arm_collides(robot, environment, np.array([theta1, theta2])):
                grid[row, col] = CELL_OBSTACLE

    # ③ 始点・終点（手先位置）を逆運動学で関節角度に直し，格子のセルへ対応づける
    cspace_map = CSpaceMap(grid, None, None, robot, environment)
    start_joint = robot.inverse_kinematics(np.array(ARM_START_POSE))
    goal_joint  = robot.inverse_kinematics(np.array(ARM_GOAL_POSE))
    start_coord = cspace_map.joint_to_coord(start_joint)
    goal_coord  = cspace_map.joint_to_coord(goal_joint)
    # 連続座標を最寄りのセルへ丸め，禁止領域なら近くの空きセルへずらす
    start_cell = _find_free_cell(grid, cspace_map.pos_to_cell(start_coord))
    goal_cell  = _find_free_cell(grid, cspace_map.pos_to_cell(goal_coord))

    # スタート・ゴールを設定した地図として作り直して返す
    return CSpaceMap(grid, start_cell, goal_cell, robot, environment)


# 第3〜10章のアルゴリズムは標準で make_sample_map を読み込む。本章ではそれを
# C-space地図づくりに読み替える（各アルゴリズムを単体実行したときの動作確認用の別名）
make_sample_map = make_cspace_map


if __name__ == "__main__":
    # 動作確認（C-space地図を作り，禁止領域の割合とスタート・ゴールを確かめる）
    cspace_map = make_cspace_map()
    obstacle_rate = cspace_map.grid.mean()
    print(f"C-space地図サイズ (行, 列) = {cspace_map.shape}")
    print(f"禁止領域（C-obstacle）の割合 = {obstacle_rate:.3f}")
    print(f"スタートセル = {cspace_map.start}, ゴールセル = {cspace_map.goal}")
    print(f"スタートの関節角度 [deg] = {np.rad2deg(cspace_map.coord_to_joint(cspace_map.cell_to_pos(cspace_map.start)))}")
