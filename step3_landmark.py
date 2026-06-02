import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import json
import os

# 경로
MODEL_PATH = r"D:\python_projects\posture\pose_landmarker.task"
CONFIG_PATH = r"D:\python_projects\posture\offsets.json"

# PoseLandmarker 초기화
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO
)
landmarker = mp_vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# ── 오프셋 (저장 파일 있으면 불러오기) ──────────────────
DEFAULT_OFFSETS = {"le": [0, 0], "re": [0, 0], "ls": [0, 0], "rs": [0, 0]}

def load_offsets():
    """offsets.json을 읽어서 반환 (없거나 오류면 기본값)."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass  # 저장 중 깨진 파일이면 이전 값 유지
    return {k: v[:] for k, v in DEFAULT_OFFSETS.items()}

offsets = load_offsets()
_last_mtime = os.path.getmtime(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else 0
print(f"시작 위치: {offsets}")

# 드래그 상태
drag_state = {"target": None, "dragging": False}
DOT_RADIUS = 8

# 점 키 목록 (귀/어깨만 — base 키와 섞이지 않게 분리)
POINT_KEYS = ["le", "re", "ls", "rs"]
points = {}       # 화면에 그려지는 실제 위치
base_points = {}  # 오프셋 적용 전 원본 랜드마크 위치


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # POINT_KEYS 안에서만 가장 가까운 점 탐색 (버그 수정)
        for key in POINT_KEYS:
            pt = points.get(key)
            if pt and abs(x - pt[0]) < DOT_RADIUS * 2 and abs(y - pt[1]) < DOT_RADIUS * 2:
                drag_state["target"] = key
                drag_state["dragging"] = True
                break

    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_state["dragging"] and drag_state["target"]:
            key = drag_state["target"]
            base = base_points.get(key)
            if base:
                offsets[key][0] = x - base[0]
                offsets[key][1] = y - base[1]

    elif event == cv2.EVENT_LBUTTONUP:
        drag_state["dragging"] = False
        drag_state["target"] = None


cv2.namedWindow("랜드마크 테스트")
cv2.setMouseCallback("랜드마크 테스트", mouse_callback)

print("드래그: 점 이동 | s: 저장 | r: 초기화 | q: 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ── 핫 리로드: offsets.json이 바뀌면 즉시 다시 읽기 ──
    # (드래그 중이 아닐 때만 — 드래그 값이 덮어써지지 않게)
    if not drag_state["dragging"] and os.path.exists(CONFIG_PATH):
        mtime = os.path.getmtime(CONFIG_PATH)
        if mtime != _last_mtime:
            offsets = load_offsets()
            _last_mtime = mtime
            print(f"[자동 반영] 위치 갱신됨: {offsets}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        def to_pixel(lm):
            return (int(lm.x * w), int(lm.y * h))

        # 원본 랜드마크 좌표 (귀 7,8 / 어깨 11,12)
        base_points["le"] = to_pixel(landmarks[7])
        base_points["re"] = to_pixel(landmarks[8])
        base_points["ls"] = to_pixel(landmarks[11])
        base_points["rs"] = to_pixel(landmarks[12])

        # 오프셋 적용한 실제 위치
        for key in POINT_KEYS:
            bx, by = base_points[key]
            points[key] = (bx + offsets[key][0], by + offsets[key][1])

        # ── 수평 맞추기: 좌우 점의 높이(y)를 평균으로 통일 ──
        ear_y = (points["le"][1] + points["re"][1]) // 2
        points["le"] = (points["le"][0], ear_y)
        points["re"] = (points["re"][0], ear_y)

        sh_y = (points["ls"][1] + points["rs"][1]) // 2
        points["ls"] = (points["ls"][0], sh_y)
        points["rs"] = (points["rs"][0], sh_y)

        # 점 그리기 (귀: 파랑, 어깨: 초록)
        cv2.circle(frame, points["le"], DOT_RADIUS, (255, 0, 0), -1)
        cv2.circle(frame, points["re"], DOT_RADIUS, (255, 0, 0), -1)
        cv2.circle(frame, points["ls"], DOT_RADIUS, (0, 255, 0), -1)
        cv2.circle(frame, points["rs"], DOT_RADIUS, (0, 255, 0), -1)

        # 안내 텍스트
        cv2.putText(frame, "drag: move | s: save | r: reset | q: quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"ear offset: {offsets['le']}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, f"shoulder offset: {offsets['ls']}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("랜드마크 테스트", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # 현재 오프셋 저장
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(offsets, f, ensure_ascii=False, indent=2)
        print(f"저장 완료! → {CONFIG_PATH}")
        print(f"   {offsets}")
    elif key == ord('r'):
        # 초기화
        offsets = {k: v[:] for k, v in DEFAULT_OFFSETS.items()}
        print("오프셋 초기화됨")

cap.release()
cv2.destroyAllWindows()
landmarker.close()
