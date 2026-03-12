import cv2
from ultralytics import YOLO
import math
import mss
import numpy as np
import ntcore

# ─── Config ───────────────────────────────────────────────────────────────────
CONF_THRES          = 0.7
REAL_DIAMETER_M     = 5.91 * 0.0254   # ball diameter in meters (fuel = ~6 in)
HORIZONTAL_FOV_DEG  = 63.5

# Priority tuning knobs
CLUSTER_RADIUS_PX   = 150   # pixel radius to consider two balls "clustered"
CLUSTER_BONUS       = 0.4   # extra weight per nearby neighbour
DISTANCE_WEIGHT     = 0.7   # how strongly distance reduces priority (higher = closer balls favoured more)

# ─── NetworkTables ────────────────────────────────────────────────────────────
# e.g. "10.TE.AM.2"  or  "roborio-XXXX-frc.local"
ROBOT_TEAM_NUMBER = 4572   # ← replace with your team number


def init_networktables() -> dict:
    """
    Start an NT4 client, connect to the roboRIO, and return a dict of
    pre-created publisher handles under BallVision/.

    Publishing as a client (not a server) means the roboRIO remains the
    NT server, which is the standard FRC setup.
    """
    inst = ntcore.NetworkTableInstance.getDefault()
    inst.startClient4("BallVisionClient")
    inst.setServerTeam(ROBOT_TEAM_NUMBER)        # resolves mDNS automatically
    # inst.setServer("10.99.99.2")               # uncomment to use a fixed IP

    table = inst.getTable("BallVision")
    return {
        "x":          table.getDoubleTopic("x").publish(),
        "y":          table.getDoubleTopic("y").publish(),
        "angle":      table.getDoubleTopic("angle").publish(),
        "has_target": table.getBooleanTopic("has_target").publish(),
    }
# CAMERA_HEIGHT_M   : vertical distance from the ground to the camera lens
# CAMERA_PITCH_DEG  : tilt of the camera below the horizontal
#                     0  = perfectly level
#                     +  = angled downward toward the floor  (typical for ball tracking)
#                     -  = angled upward
# BALL_RADIUS_M     : used to correct the contact-point vs. centre offset
CAMERA_HEIGHT_M     = 0.60    # ~24 in — replace with your actual mount height
CAMERA_PITCH_DEG    = 20.0    # degrees downward from horizontal
BALL_RADIUS_M       = REAL_DIAMETER_M / 2.0


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def focal_length(image_width: int, fov_deg: float) -> float:
    """Pixel focal length from image width and horizontal FOV."""
    return image_width / (2.0 * math.tan(math.radians(fov_deg / 2.0)))


def pixel_distance_m(bbox_width_px: int, f: float) -> float:
    """
    Slant distance from camera lens to ball centre (metres).
    Uses the apparent angular size of the ball's diameter.
    """
    if bbox_width_px == 0:
        return float("inf")
    return (REAL_DIAMETER_M * f) / bbox_width_px


def heading_angle_deg(cx_px: int, image_width: int, f: float) -> float:
    """
    Horizontal bearing from the camera optical axis to the ball (degrees).

    Positive → ball is to the RIGHT  (robot should turn right / positive yaw).
    Negative → ball is to the LEFT   (robot should turn left  / negative yaw).
    0°       → ball is dead-ahead.

    Uses the pixel offset of the ball centre from the image centre and the
    computed focal length, so it is correct regardless of resolution.
    """
    offset_px = cx_px - image_width / 2.0
    return math.degrees(math.atan2(offset_px, f))


def ground_plane_position(slant_dist_m: float,
                           bearing_deg: float,
                           vert_fov_deg: float,
                           cy_px: int,
                           image_height: int) -> tuple[float, float]:
    """
    Project a detected ball onto the ground plane and return its position
    in robot-relative coordinates (metres).

        X  =  forward distance along the floor from the camera mount point.
               Positive = ahead of the robot.
        Y  =  lateral offset.
               Positive = to the robot's RIGHT,  negative = to the LEFT.

    Method
    ------
    1. Compute the vertical ray angle (θv) from the camera's optical axis to
       the ball's pixel row, using the vertical FOV and image height.
    2. Add the camera's downward pitch (φ) so θv is now relative to horizontal.
    3. The ball's centre sits at height BALL_RADIUS_M above the floor.
       The camera is at CAMERA_HEIGHT_M.  The true depression angle to the
       ball's centre is:
           α = atan2(CAMERA_HEIGHT_M - BALL_RADIUS_M,  X_floor)
       We solve for X_floor using the combined vertical angle:
           tan(φ + θv) = (CAMERA_HEIGHT_M - BALL_RADIUS_M) / X_floor
           X_floor = (CAMERA_HEIGHT_M - BALL_RADIUS_M) / tan(φ + θv)
    4. Y is derived from X_floor and the horizontal bearing:
           Y = X_floor * tan(bearing_deg)

    Fallback: if the ray is nearly horizontal (tan ≈ 0) we fall back to the
    slant-distance projection, which is less accurate but never crashes.
    """
    phi_rad    = math.radians(CAMERA_PITCH_DEG)          # camera pitch below horizontal
    f_v        = image_height / (2.0 * math.tan(math.radians(vert_fov_deg / 2.0)))
    offset_v   = (image_height / 2.0) - cy_px            # positive = ball above centre row
    theta_v    = math.atan2(offset_v, f_v)               # vertical angle from optical axis (+up)

    # Total depression angle below horizontal to the ball centre pixel
    depression = phi_rad - theta_v                        # phi is downward, theta_v upward → subtract

    height_diff = CAMERA_HEIGHT_M - BALL_RADIUS_M        # vertical drop from lens to ball top

    tan_dep = math.tan(depression)
    if abs(tan_dep) < 1e-6:
        # Ray is nearly horizontal — fall back to slant projection
        X = slant_dist_m * math.cos(math.radians(bearing_deg))
        Y = slant_dist_m * math.sin(math.radians(bearing_deg))
    else:
        X = height_diff / tan_dep
        Y = X * math.tan(math.radians(bearing_deg))

    return X, Y


def priority_score(dist_m: float, neighbours: int) -> float:
    """
    Higher score = better target.

    Base score is inverse-square of distance (close balls are worth a lot more).
    Each detected ball within CLUSTER_RADIUS_PX of this ball adds CLUSTER_BONUS,
    rewarding paths that let the robot sweep up multiple balls in one run.
    """
    if dist_m <= 0:
        return 0.0
    base = DISTANCE_WEIGHT / (dist_m ** 2)
    return base + neighbours * CLUSTER_BONUS


# ─── Per-frame detection + selection ──────────────────────────────────────────

def detect_balls(frame, model):
    """
    Run YOLO and return a list of dicts, one per ball above CONF_THRES.

    Keys per ball:
        x1, y1, x2, y2  – bounding box corners (pixels)
        cx, cy           – bbox centre (pixels)
        conf             – YOLO confidence
        bbox_w           – bbox width (pixels)
        dist_m           – slant distance, camera → ball centre (metres)
        dist_in          – same in inches
        angle_deg        – horizontal bearing  (0° = ahead, + = right)
        robot_x          – forward distance on the ground plane (metres)
        robot_y          – lateral offset on the ground plane (metres, + = right)
    """
    results = model(frame, stream=True)
    h, w = frame.shape[:2]
    f = focal_length(w, HORIZONTAL_FOV_DEG)

    # Derive vertical FOV from horizontal FOV and pixel aspect ratio
    aspect = h / w
    vert_fov_deg = math.degrees(2.0 * math.atan(math.tan(math.radians(HORIZONTAL_FOV_DEG / 2.0)) * aspect))

    balls = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bbox_w = x2 - x1
            dist_m  = pixel_distance_m(bbox_w, f)
            bearing = heading_angle_deg(cx, w, f)
            rx, ry  = ground_plane_position(dist_m, bearing, vert_fov_deg, cy, h)

            balls.append(dict(
                x1=x1, y1=y1, x2=x2, y2=y2,
                cx=cx, cy=cy,
                conf=conf,
                bbox_w=bbox_w,
                dist_m=dist_m,
                dist_in=dist_m * 39.3701,
                angle_deg=bearing,
                robot_x=rx,
                robot_y=ry,
            ))

    return balls


def score_balls(balls):
    """
    Attach a priority score to each ball, accounting for clusters.
    Returns the same list with a 'score' key added; sorted best-first.
    """
    for i, b in enumerate(balls):
        neighbours = sum(
            1 for j, other in enumerate(balls)
            if i != j and math.hypot(b["cx"] - other["cx"], b["cy"] - other["cy"]) < CLUSTER_RADIUS_PX
        )
        b["score"]      = priority_score(b["dist_m"], neighbours)
        b["neighbours"] = neighbours

    balls.sort(key=lambda b: b["score"], reverse=True)
    return balls


def find_clusters(balls):
    """
    Group balls into clusters using union-find on CLUSTER_RADIUS_PX proximity.
    Returns a list of clusters; each cluster is a list of ball dicts.
    Only clusters with 2+ balls are returned (singletons are ignored).
    """
    n = len(balls)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            dist = math.hypot(balls[i]["cx"] - balls[j]["cx"],
                              balls[i]["cy"] - balls[j]["cy"])
            if dist < CLUSTER_RADIUS_PX:
                union(i, j)

    groups: dict[int, list] = {}
    for i, b in enumerate(balls):
        root = find(i)
        groups.setdefault(root, []).append(b)

    return [g for g in groups.values() if len(g) >= 2]


def select_target(balls, clusters):
    """
    Pick the best destination and return a unified target dict.

    Priority order:
      1. If any clusters exist, pick the cluster with the highest total score.
         Drive toward its centroid (mean robot_x / robot_y of all members),
         and use the centroid's back-projected bearing as the heading angle.
      2. Otherwise fall back to the highest-scoring individual ball.

    The returned dict always has:
        robot_x, robot_y  – ground-plane destination (metres)
        angle_deg         – horizontal bearing to that point
        score             – total score of the chosen cluster (or ball score)
        is_cluster        – True if this target is a cluster centroid
        members           – list of ball dicts in the cluster (empty for singletons)
    """
    if not balls:
        return None

    if clusters:
        # Score each cluster by summing its members' individual scores
        best_cluster = max(clusters, key=lambda g: sum(b["score"] for b in g))
        cx_mean = sum(b["cx"]      for b in best_cluster) / len(best_cluster)
        rx_mean = sum(b["robot_x"] for b in best_cluster) / len(best_cluster)
        ry_mean = sum(b["robot_y"] for b in best_cluster) / len(best_cluster)
        # Bearing to the centroid: arctan(Y / X) in robot frame
        bearing = math.degrees(math.atan2(ry_mean, rx_mean)) if rx_mean > 0 else 0.0
        return dict(
            robot_x   = rx_mean,
            robot_y   = ry_mean,
            angle_deg = bearing,
            score     = sum(b["score"] for b in best_cluster),
            is_cluster= True,
            members   = best_cluster,
            # pixel centroid for crosshair drawing
            cx        = int(sum(b["cx"] for b in best_cluster) / len(best_cluster)),
            cy        = int(sum(b["cy"] for b in best_cluster) / len(best_cluster)),
        )

    # No clusters — aim at the best individual ball
    b = balls[0]
    return dict(
        robot_x   = b["robot_x"],
        robot_y   = b["robot_y"],
        angle_deg = b["angle_deg"],
        score     = b["score"],
        is_cluster= False,
        members   = [],
        cx        = b["cx"],
        cy        = b["cy"],
    )


# ─── Drawing helpers ──────────────────────────────────────────────────────────

COLOUR_DEFAULT  = (0,   0, 255)   # BGR red   – normal ball
COLOUR_TARGET   = (0, 255,   0)   # BGR green – selected target
COLOUR_CLUSTER  = (0, 200, 255)   # BGR amber – clustered ball
FONT            = cv2.FONT_HERSHEY_SIMPLEX


def draw_heading_arrow(frame, target):
    """Draw a centred arrow showing the turn direction to the target."""
    h, w = frame.shape[:2]
    origin = (w // 2, h - 40)
    angle_rad = math.radians(target["angle_deg"])
    length = 80
    tip = (
        int(origin[0] + length * math.sin(angle_rad)),
        int(origin[1] - length * math.cos(angle_rad)),
    )
    cv2.arrowedLine(frame, origin, tip, COLOUR_TARGET, 3, tipLength=0.3)
    cv2.putText(frame, f"{target['angle_deg']:+.1f} deg", (origin[0] - 60, origin[1] + 30),
                FONT, 1.0, COLOUR_TARGET, 2)


def draw_crosshair(frame, cx, cy, size=12):
    cv2.line(frame, (cx - size, cy), (cx + size, cy), COLOUR_TARGET, 2)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), COLOUR_TARGET, 2)


COLOUR_CLUSTER_BOX = (0, 165, 255)   # BGR orange – cluster envelope
CLUSTER_BOX_PAD    = 18              # pixels of padding around the tight cluster bbox

def draw_cluster_boxes(frame, clusters):
    """
    Draw a dashed-style rounded rectangle around each cluster of 2+ balls.
    Labelled with ball count and total priority score of the group.
    """
    for group in clusters:
        # Tight bounding box over all member balls' bboxes
        gx1 = min(b["x1"] for b in group) - CLUSTER_BOX_PAD
        gy1 = min(b["y1"] for b in group) - CLUSTER_BOX_PAD
        gx2 = max(b["x2"] for b in group) + CLUSTER_BOX_PAD
        gy2 = max(b["y2"] for b in group) + CLUSTER_BOX_PAD

        # Dashed rectangle: draw as segments to differentiate from per-ball boxes
        dash_len, gap_len = 14, 7
        pts = [(gx1, gy1, gx2, gy1),   # top
               (gx2, gy1, gx2, gy2),   # right
               (gx2, gy2, gx1, gy2),   # bottom
               (gx1, gy2, gx1, gy1)]   # left

        for (ax, ay, bx, by) in pts:
            seg_len = math.hypot(bx - ax, by - ay)
            if seg_len == 0:
                continue
            dx, dy = (bx - ax) / seg_len, (by - ay) / seg_len
            pos = 0.0
            drawing = True
            while pos < seg_len:
                end_pos = min(pos + (dash_len if drawing else gap_len), seg_len)
                if drawing:
                    p1 = (int(ax + dx * pos),     int(ay + dy * pos))
                    p2 = (int(ax + dx * end_pos), int(ay + dy * end_pos))
                    cv2.line(frame, p1, p2, COLOUR_CLUSTER_BOX, 2)
                pos = end_pos
                drawing = not drawing

        # Label: ball count + summed score
        total_score = sum(b["score"] for b in group)
        label = f"{len(group)} balls  score={total_score:.2f}"
        cv2.putText(frame, label, (gx1 + 4, gy1 - 8),
                    FONT, 0.7, COLOUR_CLUSTER_BOX, 2)


def compile_image(frame, model, nt_pub: dict):
    balls    = detect_balls(frame, model)
    balls    = score_balls(balls)
    clusters = find_clusters(balls)
    target   = select_target(balls, clusters)

    # ── Publish to NetworkTables ──────────────────────────────────────────────
    if target:
        nt_pub["has_target"].set(True)
        nt_pub["x"].set(target["robot_x"])
        nt_pub["y"].set(target["robot_y"])
        nt_pub["angle"].set(target["angle_deg"])
    else:
        nt_pub["has_target"].set(False)
        nt_pub["x"].set(0.0)
        nt_pub["y"].set(0.0)
        nt_pub["angle"].set(0.0)

    # Draw cluster envelopes first (underneath per-ball boxes)
    draw_cluster_boxes(frame, clusters)

    # Highlight member balls of the chosen cluster
    chosen_members = set(id(b) for b in target["members"]) if target else set()

    for b in balls:
        in_chosen  = id(b) in chosen_members
        is_cluster = b["neighbours"] > 0
        colour    = COLOUR_TARGET if in_chosen else (COLOUR_CLUSTER if is_cluster else COLOUR_DEFAULT)
        thickness = 3 if in_chosen else 2

        cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), colour, thickness)
        cv2.putText(frame, f"{b['conf']:.2f}", (b["x1"], b["y1"] - 10),
                    FONT, 0.9, colour, 2)
        cv2.putText(frame, f"{b['dist_in']:.1f} in", (b["x1"], b["y2"] + 28),
                    FONT, 0.9, (0, 200, 0), 2)

    # Draw crosshair + position at target centroid (cluster or single ball)
    if target:
        draw_crosshair(frame, target["cx"], target["cy"])
        label = "CLUSTER" if target["is_cluster"] else "BALL"
        cv2.putText(frame, f"{label}  {target['angle_deg']:+.1f} deg", 
                    (target["cx"] - 60, target["cy"] - 20),
                    FONT, 0.8, COLOUR_TARGET, 2)
        cv2.putText(frame, f"X={target['robot_x']:.2f}m  Y={target['robot_y']:+.2f}m",
                    (target["cx"] - 60, target["cy"] - 44),
                    FONT, 0.8, COLOUR_TARGET, 2)

    # HUD: heading arrow and score bar
    if target:
        draw_heading_arrow(frame, target)
        n = len(target["members"]) if target["is_cluster"] else 1
        hud = (f"{'CLUSTER' if target['is_cluster'] else 'BALL'} ({n})  "
               f"X={target['robot_x']:.2f}m  Y={target['robot_y']:+.2f}m  "
               f"angle={target['angle_deg']:+.1f}deg  score={target['score']:.3f}  "
               f"total balls={len(balls)}")
        cv2.putText(frame, hud, (10, 36), FONT, 0.75, COLOUR_TARGET, 2)
    else:
        cv2.putText(frame, "No balls detected", (10, 36), FONT, 0.9, (100, 100, 100), 2)

    cv2.imshow("Auto-Align", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return False
    return True


# ─── Entry point ──────────────────────────────────────────────────────────────

model  = YOLO("runs/detect/train20/yolov8nboomermathballmodel.pt")
nt_pub = init_networktables()
mode   = "camera"

if mode == "camera":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam didn't open :(")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not compile_image(frame, model, nt_pub):
            break
    cap.release()
    cv2.destroyAllWindows()

elif mode == "screen":
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while True:
            screenshot = sct.grab(monitor)
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
            if not compile_image(frame, model, nt_pub):
                break