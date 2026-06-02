import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import json
import os

# ── 경로 ──────────────────────────────────────────────────────────────
MODEL_PATH  = r"D:\python_projects\posture\pose_landmarker.task"
CONFIG_PATH = r"D:\python_projects\posture\offsets.json"

# ── PoseLandmarker 초기화 (VIDEO 모드) ────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO
)
landmarker = mp_vision.PoseLandmarker.create_from_options(options)

# ── 카메라 열기 ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# ── 오프셋 로드 ────────────────────────────────────────────────────────
DEFAULT_OFFSETS = {"le": [0,0], "re": [0,0], "ls": [0,0], "rs": [0,0]}

def load_offsets():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    return {k: v[:] for k, v in DEFAULT_OFFSETS.items()}

offsets = load_offsets()
_last_mtime = os.path.getmtime(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else 0

# ── 점 관련 변수 ───────────────────────────────────────────────────────
POINT_KEYS = ["le", "re", "ls", "rs"]
points      = {}   # 화면에 그려지는 실제 위치
base_points = {}   # 오프셋 적용 전 원본 랜드마크 위치
DOT_RADIUS  = 8

drag_state = {"target": None, "dragging": False}

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for key in POINT_KEYS:
            pt = points.get(key)
            if pt and abs(x - pt[0]) < DOT_RADIUS*2 and abs(y - pt[1]) < DOT_RADIUS*2:
                drag_state["target"]   = key
                drag_state["dragging"] = True
                break
    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_state["dragging"] and drag_state["target"]:
            key  = drag_state["target"]
            base = base_points.get(key)
            if base:
                offsets[key][0] = x - base[0]
                offsets[key][1] = y - base[1]
    elif event == cv2.EVENT_LBUTTONUP:
        drag_state["dragging"] = False
        drag_state["target"]   = None

cv2.namedWindow("Posture Detect")
cv2.setMouseCallback("Posture Detect", mouse_callback)

# ── 거북목 판별 설정 ───────────────────────────────────────────────────
def _load_threshold():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f).get("threshold", 0.55)
        except (json.JSONDecodeError, ValueError):
            pass
    return 0.55

TURTLE_THRESHOLD = _load_threshold()   # offsets.json에서 불러오기

# 오탐 방지: 연속으로 N프레임 이상 거북목이어야 경고
WARN_FRAMES_NEEDED = 20   # 약 0.7초 (30fps 기준)
turtle_count   = 0        # 연속 거북목 프레임 수
is_warning     = False    # 현재 경고 중 여부

print("=" * 50)
print("4단계: 거북목 감지 시작!")
print(f"  임계값(TURTLE_THRESHOLD): {TURTLE_THRESHOLD}")
print(f"  연속감지 필요 프레임: {WARN_FRAMES_NEEDED}")
print("  t: 임계값 올리기 | y: 임계값 내리기")
print("  s: 저장 | r: 초기화 | q: 종료")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ── 핫 리로드: offsets.json 변경 감지 ──────────────────────────────
    if not drag_state["dragging"] and os.path.exists(CONFIG_PATH):
        mtime = os.path.getmtime(CONFIG_PATH)
        if mtime != _last_mtime:
            offsets     = load_offsets()
            _last_mtime = mtime
            print(f"[자동반영] {offsets}")

    # ── MediaPipe 추론 ──────────────────────────────────────────────────
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts_ms    = int(time.time() * 1000)
    result   = landmarker.detect_for_video(mp_image, ts_ms)

    ratio_text = ""   # 화면에 표시할 비율 문자열

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        def to_pixel(lm):
            return (int(lm.x * w), int(lm.y * h))

        # 원본 랜드마크 좌표 (귀 7,8 / 어깨 11,12)
        base_points["le"] = to_pixel(landmarks[7])
        base_points["re"] = to_pixel(landmarks[8])
        base_points["ls"] = to_pixel(landmarks[11])
        base_points["rs"] = to_pixel(landmarks[12])

        # 오프셋 적용
        for key in POINT_KEYS:
            bx, by = base_points[key]
            points[key] = (bx + offsets[key][0], by + offsets[key][1])

        # 수평 맞추기 (좌우 y 평균)
        ear_y = (points["le"][1] + points["re"][1]) // 2
        points["le"] = (points["le"][0], ear_y)
        points["re"] = (points["re"][0], ear_y)

        sh_y = (points["ls"][1] + points["rs"][1]) // 2
        points["ls"] = (points["ls"][0], sh_y)
        points["rs"] = (points["rs"][0], sh_y)

        # ── 거북목 판별 로직 ─────────────────────────────────────────
        # 귀-어깨 수직 거리 (픽셀)
        ear_shoulder_dist = sh_y - ear_y   # 클수록 머리가 위에 있음 (정상)

        # 어깨 너비 (픽셀) — 거리 정규화에 사용
        shoulder_width = abs(points["ls"][0] - points["rs"][0])

        if shoulder_width > 0:
            # 비율: 0.5~1.0 = 정상 / 0.5 이하 = 거북목 의심
            ratio = ear_shoulder_dist / shoulder_width
            ratio_text = f"ratio: {ratio:.2f} (기준 {TURTLE_THRESHOLD:.2f})"

            if ratio < TURTLE_THRESHOLD:
                # 거북목 의심 → 연속 카운트 증가
                turtle_count += 1
            else:
                # 정상 자세 → 카운트 초기화
                turtle_count  = 0
                is_warning    = False
        else:
            turtle_count = 0

        # 연속 N프레임 이상이면 경고 활성화
        if turtle_count >= WARN_FRAMES_NEEDED:
            is_warning = True

        # ── 점 그리기 (귀: 파랑, 어깨: 초록) ─────────────────────────
        cv2.circle(frame, points["le"], DOT_RADIUS, (255, 0,   0), -1)
        cv2.circle(frame, points["re"], DOT_RADIUS, (255, 0,   0), -1)
        cv2.circle(frame, points["ls"], DOT_RADIUS, (0,   255, 0), -1)
        cv2.circle(frame, points["rs"], DOT_RADIUS, (0,   255, 0), -1)

        # 귀-어깨 연결선 (디버그용, 얇게)
        ear_center = ((points["le"][0] + points["re"][0]) // 2, ear_y)
        sh_center  = ((points["ls"][0] + points["rs"][0]) // 2, sh_y)
        cv2.line(frame, ear_center, sh_center, (200, 200, 200), 1)

    else:
        # 랜드마크 미감지 시 카운트 초기화
        turtle_count = 0
        is_warning   = False

    # ── 경고 표시 ───────────────────────────────────────────────────────
    if is_warning:
        # 반투명 빨간 배경 (overlay)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h//2 - 60), (w, h//2 + 60), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # 경고 텍스트
        cv2.putText(frame, "! 똑바로 앉으세요 !",
                    (w//2 - 200, h//2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

    # ── 상태 정보 HUD ───────────────────────────────────────────────────
    status_color = (0, 0, 255) if is_warning else (0, 220, 0)
    status_text  = "거북목!" if is_warning else "정상"
    cv2.putText(frame, f"자세: {status_text}  ({turtle_count}/{WARN_FRAMES_NEEDED}f)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(frame, ratio_text,
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"t/y: threshold {TURTLE_THRESHOLD:.2f} | s: save | q: quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow("Posture Detect", frame)

    # ── 키 입력 ──────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_data = dict(offsets)
        save_data["threshold"] = TURTLE_THRESHOLD
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"저장 완료: offsets={offsets}, threshold={TURTLE_THRESHOLD}")
    elif key == ord('r'):
        offsets      = {k: v[:] for k, v in DEFAULT_OFFSETS.items()}
        turtle_count = 0
        is_warning   = False
        print("오프셋 초기화")
    elif key == ord('t'):
        TURTLE_THRESHOLD = round(TURTLE_THRESHOLD + 0.05, 2)
        print(f"임계값 ↑ → {TURTLE_THRESHOLD}")
    elif key == ord('y'):
        TURTLE_THRESHOLD = round(max(0.1, TURTLE_THRESHOLD - 0.05), 2)
        print(f"임계값 ↓ → {TURTLE_THRESHOLD}")

cap.release()
cv2.destroyAllWindows()
landmarker.close()
