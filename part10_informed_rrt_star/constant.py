# 経路生成（パスプランニング：障害物をよけて目的地まで進む道を計算すること）で
# 複数ファイルが共有する定数の定義
from enum import Enum


# 次元数を定義
DIMENSION_NONE  = -1    # 未定義
DIMENSION_2D    =  2    # 2次元
DIMENSION_3D    =  3    # 3次元

# 占有格子（occupancy grid：地図を格子に区切り，各セルが「空き」か「障害物」かを 0/1 で持つ地図）のセル状態
CELL_FREE     = 0    # 空きセル（通行できる）
CELL_OBSTACLE = 1    # 障害物セル（通行できない）

# 移動コスト（縦横の1手を 1.0 としたときの距離）
COST_STRAIGHT = 1.0       # 縦横（上下左右）の移動コスト
COST_DIAGONAL = 2 ** 0.5  # 斜めの移動コスト（√2 ≒ 1.414）


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


# 0割を防ぐための定数
EPSILON = 1e-6


# --- ポテンシャル法（potential field：引力と斥力の場をたどって進む経路生成）の既定パラメータ ---
K_ATTRACTIVE     = 1.0     # 引力ゲイン k_att（大きいほどゴールへ強く引かれる）
K_REPULSIVE      = 100.0   # 斥力ゲイン k_rep（大きいほど障害物から強く押し返される）
INFLUENCE_RADIUS = 3.0     # 斥力の影響半径 ρ0（この距離より遠い障害物は無視する）
STEP_GAIN        = 0.2     # 力を1ステップの移動量へ変換するゲイン
MAX_STEP_LENGTH  = 0.4     # 1ステップで進める最大距離（引力が大きい所での暴走を防ぐ）
MAX_STEPS        = 1000    # 最大反復回数（これを超えたら打ち切り）
GOAL_TOLERANCE   = 0.5     # ゴール到達とみなす距離
STUCK_THRESHOLD  = 1e-3    # この移動量を下回ったら「動けない」＝局所最小値に捕まったと判定
STUCK_PATIENCE   = 50      # この回数だけゴールへ近づけなければ局所最小値に捕まったと判定


# --- PRM（Probabilistic Roadmap：確率的ロードマップ法。自由空間にランダムな点を撒いて
#         通行可能なネットワーク〔ロードマップ〕を作り，その上で最短経路を探す手法）の既定パラメータ ---
PRM_N_SAMPLES         = 200    # 自由空間に撒くサンプル点の数 N（多いほど密だが構築が重い）
PRM_CONNECTION_RADIUS = 4.0    # 近傍接続の半径 r（この距離以内の点どうしを辺でつなごうとする）
PRM_SEGMENT_STEP      = 0.2    # 辺（線分）の衝突判定の刻み幅（小さいほど厳密だが遅い）
PRM_RANDOM_SEED       = 42     # 乱数シード（同じ結果を再現するために固定する）


# --- RRT（Rapidly-exploring Random Trees：ランダムに木を広げて経路を探す手法。
#         スタートから木を1本ずつ伸ばし，ゴール付近まで届いたら経路を復元する）の既定パラメータ ---
RRT_MAX_ITERATIONS   = 2000   # 木を伸ばす試行の最大回数（これを超えても届かなければ失敗）
RRT_STEP_SIZE        = 1.5    # 1回で枝を伸ばす距離（ステップ幅。大きいほど速いが障害物をすり抜けやすい）
RRT_GOAL_SAMPLE_RATE = 0.1    # ゴールバイアス（ゴールそのものをサンプルする確率。木をゴールへ向けやすくする）
RRT_GOAL_TOLERANCE   = 1.0    # ゴール到達とみなす距離（最新ノードがこの距離まで近づけば到達）
RRT_SEGMENT_STEP     = 0.2    # 枝（線分）の衝突判定の刻み幅（小さいほど厳密だが遅い）
RRT_RANDOM_SEED      = 42     # 乱数シード（同じ結果を再現するために固定する）


# --- RRT-Connect（双方向RRT：スタート側とゴール側の2本の木を交互に伸ばし，
#         一方の新ノードへもう一方の木を「衝突するまで一気に伸ばす〔CONNECT〕」で結ぶ手法。
#         単一RRTより速く，狭い通路にも強い）の既定パラメータ ---
RRT_CONNECT_MAX_ITERATIONS = 2000   # 木を伸ばす試行の最大回数（swap1回を1反復と数える）
RRT_CONNECT_STEP_SIZE      = 1.5    # 1回で枝を伸ばす距離（ステップ幅。RRTと揃える）
RRT_CONNECT_SEGMENT_STEP   = 0.2    # 枝（線分）の衝突判定の刻み幅（小さいほど厳密だが遅い）
RRT_CONNECT_RANDOM_SEED    = 42     # 乱数シード（同じ結果を再現するために固定する）


# extend（1ステップ伸長）が返す状態（CONNECT はこの状態を見て伸ばし続けるか決める）
class EXTEND_STATUS(Enum):
    """
    1ステップ伸長の結果（3状態）
    """
    REACHED  = 0    # 目標点まで到達した（これ以上は伸ばせない＝接続成功の候補）
    ADVANCED = 1    # ステップ幅ぶん進んだ（まだ目標点には届いていない）
    TRAPPED  = 2    # 障害物にぶつかり1歩も伸ばせなかった（行き止まり）


# --- RRT*（RRTスター：RRTに「最良の親選び〔choose parent〕」と「つなぎ直し〔rewire〕」を
#         足し，反復するほど経路コストを下げる手法。漸近最適性〔asymptotic optimality：
#         サンプルを増やすほど最適経路へ収束する性質〕を持つ）の既定パラメータ ---
RRT_STAR_MAX_ITERATIONS   = 1500   # 木を伸ばす試行の最大回数（ゴール到達後も続けてコストを下げる）
RRT_STAR_STEP_SIZE        = 1.5    # 1回で枝を伸ばす距離（ステップ幅。RRTと揃える）
RRT_STAR_GOAL_SAMPLE_RATE = 0.05   # ゴールバイアス（ゴールそのものをサンプルする確率。最適化のため低めにする）
RRT_STAR_GOAL_TOLERANCE   = 1.0    # ゴール到達とみなす距離（ノードがこの距離まで近づき直進できれば到達）
RRT_STAR_SEGMENT_STEP     = 0.2    # 枝（線分）の衝突判定の刻み幅（小さいほど厳密だが遅い）
RRT_STAR_RANDOM_SEED      = 42     # 乱数シード（同じ結果を再現するために固定する）

# 近傍半径 r_n = γ・(log n / n)^(1/d)（n＝ノード数，d＝次元）。ノードが増えるほど半径を縮め，
#   choose parent / rewire の対象を絞りながら漸近最適性を保つ
RRT_STAR_NEIGHBOR_GAMMA   = 18.0   # 近傍半径の係数 γ（大きいほど広く近傍を探す＝最適化が進むが重い）
RRT_STAR_NEIGHBOR_MAX     = 5.0    # 近傍半径の上限（序盤に半径が大きくなりすぎるのを防ぐ頭打ち）


# --- Informed RRT*（インフォームドRRTスター：RRT* に「初期解が見つかった後はサンプリング領域を
#         楕円体〔ellipsoid〕に絞る」工夫を足し，無駄なサンプルを捨てて最適化を加速する手法。
#         スタートとゴールを焦点とし，現在の最良経路長 c_best を長径とする楕円体の内側だけ探す。
#         この楕円体は「2焦点からの距離の和が c_best 以下」の点の集まりで，
#         最良経路を縮められる可能性のある点はすべてこの中にある）の既定パラメータ ---
INFORMED_MAX_ITERATIONS   = 1500   # 木を伸ばす試行の最大回数（RRT* と揃えて収束の速さを公平に比べる）
INFORMED_STEP_SIZE        = 1.5    # 1回で枝を伸ばす距離（ステップ幅。RRT* と揃える）
INFORMED_GOAL_SAMPLE_RATE = 0.05   # ゴールバイアス（ゴールそのものをサンプルする確率。RRT* と揃える）
INFORMED_GOAL_TOLERANCE   = 1.0    # ゴール到達とみなす距離（ノードがこの距離まで近づき直進できれば到達）
INFORMED_SEGMENT_STEP     = 0.2    # 枝（線分）の衝突判定の刻み幅（小さいほど厳密だが遅い）
INFORMED_NEIGHBOR_GAMMA   = 18.0   # 近傍半径の係数 γ（RRT* と揃える）
INFORMED_NEIGHBOR_MAX     = 5.0    # 近傍半径の上限（RRT* と揃える）
INFORMED_RANDOM_SEED      = 42     # 乱数シード（同じ結果を再現するために固定する）
