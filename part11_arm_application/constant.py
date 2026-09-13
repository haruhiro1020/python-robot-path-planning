# 第11章（ロボットアームへの応用）で複数ファイルが共有する定数の定義
#
# 第1〜10章は2次元グリッド上の「点ロボット」を動かしてきました。本章では同じ8手法を
# 2軸ロボットアームの「関節空間（joint space, C-space：関節角度 (θ1, θ2) を座標とする空間。
# アームの姿勢1つが、この空間の1点に対応する）」へ載せ替えます。
#
# ポイントは「点ロボットのコードをほぼそのまま使う」こと。関節空間を格子に区切れば，
# 第3〜10章の地図・探索・サンプリングのコードがそのまま動きます（座標 (x, y) を関節角度
# (θ1, θ2) と読み替えるだけ）。違うのは衝突判定だけで，点が障害物セルに入ったかではなく，
# 「その関節角度のときアームが障害物にぶつかるか」を python-fcl で判定します（collision.py）。

from enum import Enum
from enum import auto


# 次元数を定義
DIMENSION_NONE  = -1    # 未定義
DIMENSION_2D    =  2    # 2次元
DIMENSION_3D    =  3    # 3次元

# 占有格子（occupancy grid：地図を格子に区切り，各セルが「空き」か「障害物」かを 0/1 で持つ地図）のセル状態
# 本章では「関節空間を格子に区切った地図（C-space地図）」のセル状態として使う
CELL_FREE     = 0    # 空きセル（アームがぶつからない関節角度）
CELL_OBSTACLE = 1    # 障害物セル（アームが障害物にぶつかる関節角度＝禁止領域 C-obstacle）

# 移動コスト（縦横の1手を 1.0 としたときの距離）
COST_STRAIGHT = 1.0       # 縦横（上下左右）の移動コスト
COST_DIAGONAL = 2 ** 0.5  # 斜めの移動コスト（√2 ≒ 1.414）

# 0割を防ぐための定数
EPSILON = 1e-6


# 近傍（あるセルから1手で移動できる隣接セル）の種類を定義
class CONNECTIVITY(Enum):
    """
    近傍の種類（4近傍：上下左右のみ／8近傍：斜めも含む）
    """
    FOUR  = 4    # 4近傍（上下左右）
    EIGHT = 8    # 8近傍（上下左右＋斜め）


# ヒューリスティック（heuristic：ゴールまでのおおよその距離を見積もる関数。探索をゴール方向へ誘導する）の種類
class HEURISTIC(Enum):
    """
    ヒューリスティックの種類（ゴールまでの距離をどう見積もるか）
    """
    MANHATTAN = 0    # マンハッタン距離（縦横の移動量の和。|Δrow| + |Δcol|）
    EUCLIDEAN = 1    # ユークリッド距離（まっすぐ測った直線距離。√(Δrow² + Δcol²)）


# --- 回転軸（robot.py / rotation.py が使う。前作 src/two_dof/part4_rrt_fcl と同じ） ---
ROTATION_X_AXIS = "rot_x"   # x軸周りに回転
ROTATION_Y_AXIS = "rot_y"   # y軸周りに回転
ROTATION_Z_AXIS = "rot_z"   # z軸周りに回転


# 干渉物（障害物）の名称を定義（environment.py / plot.py / animation.py が使う）
class INTERFERENCE(Enum):
    """
    干渉物の名称を定義（描画と干渉判定で形ごとに扱いを分けるための識別子）
    """
    NONE      = ""           # 未定義
    CIRCLE    = "circle"     # 円形の干渉物
    RECTANGLE = "rectangle"  # 長方形の干渉物


# --- 関節空間（C-space）の設定 ---
# 2軸アームの関節空間は (θ1, θ2) の2次元。各関節を −π〜π [rad] の範囲で動かす
import math
JOINT_MIN = -math.pi   # 各関節の最小角度 [rad]（−π）
JOINT_MAX =  math.pi   # 各関節の最大角度 [rad]（+π）
# 関節空間を格子に区切るときの分割数（1辺のセル数）。72分割＝5°刻み（細かいほど地図は精密だが重い）
CSPACE_RESOLUTION = 72

# 干渉判定のマージン [m]（アームと障害物の最短距離がこの値以下なら「ぶつかる」とみなす。
#   0.0 なら実際に重なったときだけ衝突。少し大きくすると安全側の余裕を持たせられる）
COLLISION_MARGIN = 0.0


# 2軸アームのリンク長 [m]（リンク1・リンク2とも 1.0 m。前作と同じ）
ARM_LINK_LENGTHS = (1.0, 1.0)
# 経路生成の始点・終点（アームの手先位置 [m]）。逆運動学で関節角度へ変換してから関節空間で探索する
ARM_START_POSE = (1.0, -1.0)   # 始点の手先位置 (x, y) [m]
ARM_GOAL_POSE  = (1.0,  1.0)   # 終点の手先位置 (x, y) [m]


# === 以下，各経路生成手法の既定パラメータ ===
# 第1〜10章は 20×20 のグリッドだったが，本章の C-space地図は 72×72 とひとまわり大きい。
# そのぶん「1歩で進む距離」や「近傍半径」などの長さに関わる値を，グリッドの広さに合わせて
# 大きめに取り直している（手法そのものは第3〜10章と同一）。

# --- ポテンシャル法（potential field：引力と斥力の場をたどって進む経路生成）の既定パラメータ ---
K_ATTRACTIVE     = 1.0     # 引力ゲイン k_att（大きいほどゴールへ強く引かれる）
K_REPULSIVE      = 200.0   # 斥力ゲイン k_rep（大きいほど禁止領域から強く押し返される）
INFLUENCE_RADIUS = 3.0     # 斥力の影響半径 ρ0（このセル距離より遠い禁止領域は無視する）
STEP_GAIN        = 0.3     # 力を1ステップの移動量へ変換するゲイン
MAX_STEP_LENGTH  = 0.8     # 1ステップで進める最大距離（引力が大きい所での暴走を防ぐ）
MAX_STEPS        = 2000    # 最大反復回数（これを超えたら打ち切り）
GOAL_TOLERANCE   = 2.0     # ゴール到達とみなすセル距離
STUCK_THRESHOLD  = 1e-3    # この移動量を下回ったら「動けない」＝局所最小値に捕まったと判定
STUCK_PATIENCE   = 100     # この回数だけゴールへ近づけなければ局所最小値に捕まったと判定


# --- PRM（Probabilistic Roadmap：確率的ロードマップ法。自由空間にランダムな点を撒いて
#         通行可能なネットワーク〔ロードマップ〕を作り，その上で最短経路を探す手法）の既定パラメータ ---
PRM_N_SAMPLES         = 600    # 自由空間（ぶつからない関節角度）に撒くサンプル点の数 N
PRM_CONNECTION_RADIUS = 8.0    # 近傍接続の半径 r（このセル距離以内の点どうしを辺でつなごうとする）
PRM_SEGMENT_STEP      = 0.3    # 辺（線分）の衝突判定の刻み幅（小さいほど厳密だが遅い）
PRM_RANDOM_SEED       = 42     # 乱数シード（同じ結果を再現するために固定する）


# --- RRT（Rapidly-exploring Random Trees：ランダムに木を広げて経路を探す手法）の既定パラメータ ---
RRT_MAX_ITERATIONS   = 4000   # 木を伸ばす試行の最大回数（これを超えても届かなければ失敗）
RRT_STEP_SIZE        = 3.0    # 1回で枝を伸ばす距離（ステップ幅。72分割の地図に合わせて大きめ）
RRT_GOAL_SAMPLE_RATE = 0.1    # ゴールバイアス（ゴールそのものをサンプルする確率）
RRT_GOAL_TOLERANCE   = 2.0    # ゴール到達とみなすセル距離
RRT_SEGMENT_STEP     = 0.3    # 枝（線分）の衝突判定の刻み幅（小さいほど厳密だが遅い）
RRT_RANDOM_SEED      = 42     # 乱数シード（同じ結果を再現するために固定する）


# --- RRT-Connect（双方向RRT）の既定パラメータ ---
RRT_CONNECT_MAX_ITERATIONS = 4000   # 木を伸ばす試行の最大回数（swap1回を1反復と数える）
RRT_CONNECT_STEP_SIZE      = 3.0    # 1回で枝を伸ばす距離（ステップ幅。RRTと揃える）
RRT_CONNECT_SEGMENT_STEP   = 0.3    # 枝（線分）の衝突判定の刻み幅
RRT_CONNECT_RANDOM_SEED    = 42     # 乱数シード（同じ結果を再現するために固定する）


# extend（1ステップ伸長）が返す状態（CONNECT はこの状態を見て伸ばし続けるか決める）
class EXTEND_STATUS(Enum):
    """
    1ステップ伸長の結果（3状態）
    """
    REACHED  = 0    # 目標点まで到達した（これ以上は伸ばせない＝接続成功の候補）
    ADVANCED = 1    # ステップ幅ぶん進んだ（まだ目標点には届いていない）
    TRAPPED  = 2    # 障害物にぶつかり1歩も伸ばせなかった（行き止まり）


# --- RRT*（RRTスター：RRTに choose parent と rewire を足し，反復するほど経路を短くする手法）の既定パラメータ ---
RRT_STAR_MAX_ITERATIONS   = 2000   # 木を伸ばす試行の最大回数（ゴール到達後も続けてコストを下げる）
RRT_STAR_STEP_SIZE        = 3.0    # 1回で枝を伸ばす距離（ステップ幅。RRTと揃える）
RRT_STAR_GOAL_SAMPLE_RATE = 0.05   # ゴールバイアス（最適化のため低めにする）
RRT_STAR_GOAL_TOLERANCE   = 2.0    # ゴール到達とみなすセル距離
RRT_STAR_SEGMENT_STEP     = 0.3    # 枝（線分）の衝突判定の刻み幅
RRT_STAR_RANDOM_SEED      = 42     # 乱数シード（同じ結果を再現するために固定する）

# 近傍半径 r_n = γ・(log n / n)^(1/d)。72分割の地図に合わせて γ・上限を大きめに取る
# （上限を大きくしすぎると近傍が増えて rewire の衝突判定が重くなるため，ほどほどに抑える）
RRT_STAR_NEIGHBOR_GAMMA   = 35.0   # 近傍半径の係数 γ（大きいほど広く近傍を探す＝最適化が進むが重い）
RRT_STAR_NEIGHBOR_MAX     = 6.0    # 近傍半径の上限（序盤に半径が大きくなりすぎるのを防ぐ頭打ち）


# --- Informed RRT*（インフォームドRRTスター：初期解が出た後はサンプリング領域を楕円体に絞る手法）の既定パラメータ ---
INFORMED_MAX_ITERATIONS   = 2000   # 木を伸ばす試行の最大回数（RRT* と揃えて収束の速さを公平に比べる）
INFORMED_STEP_SIZE        = 3.0    # 1回で枝を伸ばす距離（ステップ幅。RRT* と揃える）
INFORMED_GOAL_SAMPLE_RATE = 0.05   # ゴールバイアス（RRT* と揃える）
INFORMED_GOAL_TOLERANCE   = 2.0    # ゴール到達とみなすセル距離
INFORMED_SEGMENT_STEP     = 0.3    # 枝（線分）の衝突判定の刻み幅
INFORMED_NEIGHBOR_GAMMA   = 35.0   # 近傍半径の係数 γ（RRT* と揃える）
INFORMED_NEIGHBOR_MAX     = 6.0    # 近傍半径の上限（RRT* と揃える）
INFORMED_RANDOM_SEED      = 42     # 乱数シード（同じ結果を再現するために固定する）
