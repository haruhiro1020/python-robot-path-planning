# 占有格子地図（点ロボットが動く2次元グリッド）のクラスと，本書共通のサンプル地図

# ライブラリの読み込み
import numpy as np

# 自作モジュールの読み込み
from constant import *      # 定数


class GridMap:
    """
    占有格子地図クラス（2次元グリッドを 0=空き / 1=障害物 で表す地図）

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


def make_sample_map() -> GridMap:
    """
    本書で共通して使うサンプル地図を作成（壁・部屋・狭い通路を含む）

    戻り値
        grid_map: サンプルの占有格子地図（スタート・ゴール設定済み）
    """
    rows, cols = 20, 20
    grid = np.zeros((rows, cols), dtype=int)

    # 縦の壁（中央付近）に，狭い通路を1か所だけ空ける
    grid[2:15, 10] = CELL_OBSTACLE
    grid[8, 10]    = CELL_FREE       # 通路（隙間）

    # 横の壁（下側に部屋を作る）
    grid[15, 3:12] = CELL_OBSTACLE

    # 箱型の障害物
    grid[4:8, 3:6] = CELL_OBSTACLE

    # スタート（左下）とゴール（右上）
    start = (18, 1)
    goal  = (1, 18)

    return GridMap(grid, start, goal)
