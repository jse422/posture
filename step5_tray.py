import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import json
import os
import threading
import tkinter as tk
from PIL import Image, ImageTk
from datetime import datetime
import joblib

# scikit-learn은 train_model_from_csv() 안에서만 사용하므로 여기서는 임포트하지 않음
# (프로그램 시작 속도를 빠르게 유지하기 위해)

# ── 경로 ──────────────────────────────────────────────────────────────
MODEL_PATH     = r"D:\python_projects\posture\pose_landmarker.task"
CONFIG_PATH    = r"D:\python_projects\posture\offsets.json"
IMAGE_PATH     = r"D:\python_projects\posture\bunny-rabbit.gif"
CSV_PATH       = r"D:\python_projects\posture\posture_log.csv"
DATASET_PATH   = r"D:\python_projects\posture\posture_dataset.csv"
MODEL_PKL_PATH = r"D:\python_projects\posture\posture_model.pkl"

# ── 거북목 판별 기본 설정 ──────────────────────────────────────────────
TURTLE_THRESHOLD   = 0.55   # 모델 없을 때 사용하는 고정 임계값
WARN_FRAMES_NEEDED = 20     # 연속 N프레임 bad 감지 시 팝업 경고
RECOVER_FRAMES     = 30     # 정상 자세 N프레임 유지 시 팝업 닫힘

# ── 데이터 수집 설정 ──────────────────────────────────────────────────
COLLECT_DURATION = 10    # 수집 시간(초)
COLLECT_INTERVAL = 0.2   # 수집 간격(초) — 초당 5샘플

# ── feature 컬럼 이름 목록 ────────────────────────────────────────────
# 이 순서를 반드시 유지해야 학습/예측 결과가 일치함
FEATURE_COLS = [
    "ratio",            # 귀~어깨 거리 / 어깨 너비
    "ear_shoulder_dist",# 어깨 평균 y - 귀 평균 y (픽셀)
    "shoulder_width",   # 좌우 어깨 x 거리 (픽셀)
    "head_center_x",    # 양쪽 귀 x 평균
    "shoulder_center_x",# 양쪽 어깨 x 평균
    "head_offset_x",    # head_center_x - shoulder_center_x (좌우 쏠림)
    "shoulder_slope",   # 오른쪽 어깨 y - 왼쪽 어깨 y (어깨 기울기)
    "ear_slope",        # 오른쪽 귀 y - 왼쪽 귀 y (고개 기울기)
    "left_ear_y",
    "right_ear_y",
    "left_shoulder_y",
    "right_shoulder_y",
]

# ── 공유 상태 (스레드 간 통신) ─────────────────────────────────────────
state = {
    "is_turtle": False,
    "popup_open": False,
    "running":    True
}
state_lock = threading.Lock()

# ────────────────────────────────────────────────────────────────────
# 오프셋 / threshold 로드·저장
# ────────────────────────────────────────────────────────────────────

DEFAULT_OFFSETS = {"le": [0,0], "re": [0,0], "ls": [0,0], "rs": [0,0]}

def load_offsets():
    """offsets.json을 읽어 전체 딕셔너리를 반환한다."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    return {k: v[:] for k, v in DEFAULT_OFFSETS.items()}

def load_threshold():
    """저장된 threshold 값을 읽는다. 없으면 기본값 반환."""
    return load_offsets().get("threshold", TURTLE_THRESHOLD)

def save_threshold(threshold, offsets):
    """threshold를 offsets.json에 저장한다. 기존 데이터는 유지된다."""
    data = load_offsets()
    data["threshold"] = threshold
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"저장 완료! threshold={threshold}")

# ────────────────────────────────────────────────────────────────────
# feature 추출
# ────────────────────────────────────────────────────────────────────

def extract_features(points):
    """le/re/ls/rs 좌표에서 12개 feature를 계산해 dict로 반환한다.
    어깨 너비가 0이거나 좌표가 없으면 None을 반환한다."""
    try:
        le_x, le_y = points["le"]
        re_x, re_y = points["re"]
        ls_x, ls_y = points["ls"]
        rs_x, rs_y = points["rs"]

        ear_y            = (le_y + re_y) / 2
        sh_y             = (ls_y + rs_y) / 2
        ear_shoulder_dist = sh_y - ear_y
        shoulder_width   = abs(ls_x - rs_x)

        if shoulder_width == 0:
            return None  # 어깨 너비가 0이면 ratio 계산 불가

        ratio            = ear_shoulder_dist / shoulder_width
        head_center_x    = (le_x + re_x) / 2
        shoulder_center_x = (ls_x + rs_x) / 2
        head_offset_x    = head_center_x - shoulder_center_x
        shoulder_slope   = rs_y - ls_y   # 양수: 오른쪽 어깨가 아래
        ear_slope        = re_y - le_y   # 양수: 오른쪽 귀가 아래

        return {
            "ratio":             round(ratio, 4),
            "ear_shoulder_dist": round(ear_shoulder_dist, 2),
            "shoulder_width":    round(shoulder_width, 2),
            "head_center_x":     round(head_center_x, 2),
            "shoulder_center_x": round(shoulder_center_x, 2),
            "head_offset_x":     round(head_offset_x, 2),
            "shoulder_slope":    round(shoulder_slope, 2),
            "ear_slope":         round(ear_slope, 2),
            "left_ear_y":        round(le_y, 2),
            "right_ear_y":       round(re_y, 2),
            "left_shoulder_y":   round(ls_y, 2),
            "right_shoulder_y":  round(rs_y, 2),
        }
    except (KeyError, TypeError):
        return None

# ────────────────────────────────────────────────────────────────────
# 데이터셋 CSV 함수
# ────────────────────────────────────────────────────────────────────

def init_dataset_csv():
    """posture_dataset.csv 파일이 없으면 헤더를 1회 생성한다."""
    if not os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + FEATURE_COLS + ["label"])
        print(f"데이터셋 파일 생성: {DATASET_PATH}")

def append_sample_to_csv(features, label):
    """feature dict와 label(good/bad)을 posture_dataset.csv에 한 줄 추가한다."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp] + [features[col] for col in FEATURE_COLS] + [label]
    with open(DATASET_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

# ────────────────────────────────────────────────────────────────────
# 실시간 로그 CSV 함수
# ────────────────────────────────────────────────────────────────────

def init_log_csv():
    """posture_log.csv 파일이 없으면 헤더를 1회 생성한다."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "ratio", "baseline_ratio", "difference", "prediction", "mode"])

def log_to_csv(ratio, baseline_ratio, difference, prediction, mode):
    """현재 자세 상태를 posture_log.csv에 1초마다 기록한다."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp,
                         f"{ratio:.4f}"          if ratio          is not None else "",
                         f"{baseline_ratio:.4f}" if baseline_ratio is not None else "",
                         f"{difference:.4f}"     if difference     is not None else "",
                         prediction,
                         mode])

# ────────────────────────────────────────────────────────────────────
# ML 모델 학습 / 로드
# ────────────────────────────────────────────────────────────────────

def train_model_from_csv():
    """posture_dataset.csv를 읽어 RandomForestClassifier를 학습하고 저장한다.
    good/bad 각각 30개 미만이면 학습을 중단한다."""
    # sklearn은 학습 시에만 임포트 (시작 속도 유지)
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    if not os.path.exists(DATASET_PATH):
        print("데이터셋 파일이 없습니다. G/B 키로 먼저 데이터를 수집하세요.")
        return

    df = pd.read_csv(DATASET_PATH)

    # 라벨별 샘플 수 확인
    counts   = df["label"].value_counts()
    bad_cols = [l for l in counts.index if l.startswith("bad")]
    bad_total = sum(counts.get(l, 0) for l in bad_cols)
    print(f"[학습] good: {counts.get('good', 0)}개, bad 합계: {bad_total}개")
    for lbl in bad_cols:
        print(f"       {lbl}: {counts.get(lbl, 0)}개")

    if counts.get("good", 0) < 10 or bad_total < 10:
        print("학습 불가: good 10개 이상 + bad 계열 합계 10개 이상 필요합니다.")
        print("  G: 좋은 자세  1: 앞으로  2: 왼쪽  3: 오른쪽  4: 등굽힘")
        return

    X = df[FEATURE_COLS].values
    y = df["label"].values

    # stratify: 클래스당 샘플이 너무 적으면 에러가 나므로 실패 시 stratify 없이 재시도
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print("일부 클래스 샘플이 적어 stratify 없이 분할합니다.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n[분류 결과]")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, MODEL_PKL_PATH)
    print(f"모델 저장 완료: {MODEL_PKL_PATH}")

def load_model_if_exists():
    """저장된 posture_model.pkl이 있으면 로드해서 반환한다. 없으면 None 반환."""
    if os.path.exists(MODEL_PKL_PATH):
        try:
            clf = joblib.load(MODEL_PKL_PATH)
            print(f"모델 로드 완료: {MODEL_PKL_PATH}")
            return clf
        except Exception as e:
            print(f"모델 로드 실패: {e}")
    return None

# ────────────────────────────────────────────────────────────────────
# 카메라 + 자세 감지 스레드
# ────────────────────────────────────────────────────────────────────

def camera_thread():
    """카메라에서 프레임을 읽고 MediaPipe로 랜드마크를 추출한다.
    모델이 있으면 ML 예측, 없으면 threshold 방식으로 거북목을 판단한다.
    G/B/T/M/PgUp/PgDn/S/Q 키 입력을 처리한다."""

    # MediaPipe 초기화
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 설정값 로드 — load_offsets()를 한 번만 호출해 파일 중복 읽기 방지
    offsets        = load_offsets()
    threshold      = offsets.get("threshold", TURTLE_THRESHOLD)
    baseline_ratio = offsets.get("baseline_ratio", None)

    # ML 모델 로드 (있으면 자동으로 MODEL 모드 시작)
    model          = load_model_if_exists()
    use_model_mode = model is not None

    POINT_KEYS = ["le", "re", "ls", "rs"]
    points      = {}
    base_points = {}

    # 거북목 판단 카운터
    turtle_count = 0
    good_count   = 0

    # 데이터 수집 상태
    collecting_label   = None   # "good" | "bad" | None
    collect_start_time = 0.0
    last_collect_time  = 0.0

    # CSV 초기화
    init_dataset_csv()
    init_log_csv()
    last_log_time = time.time()

    # 미리보기 창
    cv2.namedWindow("Posture Detect (Running)")
    cv2.resizeWindow("Posture Detect (Running)", 240, 180)
    cv2.moveWindow("Posture Detect (Running)", 20, 20)

    print("실행 중...")
    print("  G: 좋은 자세 수집")
    print("  1: 앞으로  2: 왼쪽  3: 오른쪽  4: 등굽힘  (나쁜 자세 수집)")
    print("  T: 모델 학습  M: 모드 전환  PgUp/PgDn: threshold  S: 저장  Q: 종료")

    def to_pixel(lm, w, h):
        """정규화 좌표(0~1)를 픽셀 좌표로 변환한다. 루프 밖에서 한 번만 정의."""
        return (int(lm.x * w), int(lm.y * h))

    # 학습 스레드 참조 — T 키 중복 실행 방지용
    train_thread = None

    while True:
        with state_lock:
            if not state["running"]:
                break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # MediaPipe 포즈 추출
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms    = int(time.time() * 1000)
        result   = landmarker.detect_for_video(mp_image, ts_ms)

        # 매 루프마다 초기화 — 랜드마크 미감지 시 None 유지
        ratio         = None
        features      = None
        prediction    = "?"
        is_turtle_now = False

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # 귀(7,8) · 어깨(11,12) 픽셀 좌표 추출
            base_points["le"] = to_pixel(landmarks[7],  w, h)
            base_points["re"] = to_pixel(landmarks[8],  w, h)
            base_points["ls"] = to_pixel(landmarks[11], w, h)
            base_points["rs"] = to_pixel(landmarks[12], w, h)

            # 사용자별 오프셋 적용
            for key in POINT_KEYS:
                bx, by = base_points[key]
                points[key] = (bx + offsets[key][0], by + offsets[key][1])

            # 귀·어깨 y좌표를 좌우 평균으로 통일 (기울어진 자세 보정)
            ear_y = (points["le"][1] + points["re"][1]) // 2
            points["le"] = (points["le"][0], ear_y)
            points["re"] = (points["re"][0], ear_y)
            sh_y = (points["ls"][1] + points["rs"][1]) // 2
            points["ls"] = (points["ls"][0], sh_y)
            points["rs"] = (points["rs"][0], sh_y)

            # 12개 feature 계산
            features = extract_features(points)

            if features is not None:
                ratio = features["ratio"]

                # ── 데이터 수집 (G/B 키로 시작된 경우) ──
                if collecting_label:
                    elapsed = time.time() - collect_start_time
                    if elapsed < COLLECT_DURATION:
                        if (time.time() - last_collect_time) >= COLLECT_INTERVAL:
                            append_sample_to_csv(features, collecting_label)
                            last_collect_time = time.time()
                    else:
                        print(f"[{collecting_label}] 수집 완료")
                        collecting_label = None

                # ── 자세 판별 ──
                if use_model_mode and model is not None:
                    # ML 모델 예측: feature 값을 리스트로 변환해 예측
                    feature_values = [features[col] for col in FEATURE_COLS]
                    prediction = model.predict([feature_values])[0]  # "good" or "bad_*"
                    # bad_forward / bad_left / bad_right / bad_slouch 모두 나쁜 자세로 판정
                    if prediction.startswith("bad"):
                        turtle_count += 1
                        good_count    = 0
                    else:
                        good_count   += 1
                        turtle_count  = 0
                else:
                    # 모델 없을 때: 기존 ratio/threshold 방식 fallback
                    prediction = "bad_forward" if ratio < threshold else "good"
                    if ratio < threshold:
                        turtle_count += 1
                        good_count    = 0
                    else:
                        good_count   += 1
                        turtle_count  = 0

        else:
            # 랜드마크 미감지 시 카운터 리셋
            turtle_count = 0
            good_count  += 1

        is_turtle_now = turtle_count >= WARN_FRAMES_NEEDED

        # ── 공유 상태 업데이트 ──
        with state_lock:
            state["is_turtle"] = is_turtle_now
            if good_count >= RECOVER_FRAMES:
                state["is_turtle"] = False

        # ── 실시간 로그 (1초마다) ──
        now = time.time()
        if ratio is not None and (now - last_log_time) >= 1.0:
            mode_str   = "MODEL" if use_model_mode else "THRESH"
            difference = round(ratio - baseline_ratio, 4) if baseline_ratio is not None else None
            log_to_csv(ratio, baseline_ratio, difference, prediction, mode_str)
            last_log_time = now

        # ── 미리보기 창 그리기 ──
        small = cv2.resize(frame, (240, 180))

        # 상단 상태 바 (good=초록, bad=빨강)
        bar_color = (0, 0, 220) if is_turtle_now else (0, 180, 0)
        cv2.rectangle(small, (0, 0), (240, 20), bar_color, -1)

        mode_label = "MODEL" if use_model_mode else "THRESH"
        if ratio is not None:
            status_text = f"[{mode_label}] {prediction}  r:{ratio:.2f}"
        else:
            status_text = "감지 안됨"
        cv2.putText(small, status_text, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)

        # 데이터 수집 중 표시
        if collecting_label:
            elapsed = time.time() - collect_start_time
            remain  = max(0, COLLECT_DURATION - elapsed)
            cv2.rectangle(small, (0, 22), (240, 44), (180, 100, 0), -1)
            cv2.putText(small, f"Collecting {collecting_label.upper()} {remain:.0f}s",
                        (4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)

        # 모델 없을 때 안내 메시지
        if not use_model_mode or model is None:
            cv2.putText(small, "G:good 1:fwd 2:left 3:right 4:slouch",
                        (4, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200,200,100), 1)
            cv2.putText(small, "T:train model",
                        (4, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200,200,100), 1)

        # threshold 표시 (THRESH 모드일 때)
        if not use_model_mode:
            cv2.putText(small, f"threshold: {threshold:.2f}  PgUp/Dn",
                        (4, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180,180,180), 1)

        cv2.imshow("Posture Detect (Running)", small)

        # ── 키 입력 처리 ──
        # waitKey(1)에서 특수키는 상위 비트에 코드가 들어오므로 전체 값을 확인
        raw_key = cv2.waitKey(1)
        key     = raw_key & 0xFF  # 일반 문자 키

        if key == ord('q') or cv2.getWindowProperty("Posture Detect (Running)", cv2.WND_PROP_VISIBLE) < 1:
            with state_lock:
                state["running"] = False
            break

        elif key == ord('g'):
            collecting_label   = "good"
            collect_start_time = time.time()
            print(f"좋은 자세 수집 시작 ({COLLECT_DURATION}초) - 바른 자세를 유지해주세요")

        elif key == ord('1'):
            # 앞으로 숙인 자세 (거북목)
            collecting_label   = "bad_forward"
            collect_start_time = time.time()
            print(f"나쁜 자세[앞으로] 수집 시작 ({COLLECT_DURATION}초)")

        elif key == ord('2'):
            # 왼쪽으로 기운 자세
            collecting_label   = "bad_left"
            collect_start_time = time.time()
            print(f"나쁜 자세[왼쪽] 수집 시작 ({COLLECT_DURATION}초)")

        elif key == ord('3'):
            # 오른쪽으로 기운 자세
            collecting_label   = "bad_right"
            collect_start_time = time.time()
            print(f"나쁜 자세[오른쪽] 수집 시작 ({COLLECT_DURATION}초)")

        elif key == ord('4'):
            # 등 굽힘 자세
            collecting_label   = "bad_slouch"
            collect_start_time = time.time()
            print(f"나쁜 자세[등굽힘] 수집 시작 ({COLLECT_DURATION}초)")

        elif key == ord('t'):
            # 이미 학습 중이면 중복 실행 방지
            if train_thread and train_thread.is_alive():
                print("이미 학습 중입니다. 잠시 기다려주세요.")
            else:
                def _train_and_reload():
                    """학습을 백그라운드 스레드에서 실행해 카메라가 멈추지 않게 한다."""
                    nonlocal model, use_model_mode
                    train_model_from_csv()
                    model          = load_model_if_exists()
                    use_model_mode = model is not None
                train_thread = threading.Thread(target=_train_and_reload, daemon=True)
                train_thread.start()
                print("모델 학습 시작 (백그라운드)...")

        elif key == ord('m'):
            # 모델 예측 / threshold 모드 전환
            if model is not None:
                use_model_mode = not use_model_mode
                print(f"모드 전환: {'MODEL' if use_model_mode else 'THRESHOLD'}")
            else:
                print("모델이 없습니다. T 키로 먼저 학습하세요.")

        elif key == ord('s'):
            # threshold 저장 후 baseline_ratio 메모리 갱신
            save_threshold(threshold, offsets)
            baseline_ratio = load_offsets().get("baseline_ratio", None)

        else:
            # PgUp / PgDn 특수키 처리 (Windows OpenCV)
            # raw_key를 그대로 사용해 상위 비트 포함 확인
            if raw_key == 0x210000 or raw_key == 2162688:   # PgUp
                threshold = round(threshold + 0.05, 2)
                print(f"threshold ↑ {threshold}")
            elif raw_key == 0x220000 or raw_key == 2228224:  # PgDn
                threshold = round(max(0.1, threshold - 0.05), 2)
                print(f"threshold ↓ {threshold}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("카메라 종료")

# ────────────────────────────────────────────────────────────────────
# 팝업 관리 (tkinter, 메인 스레드에서 호출) — 기존 코드 유지
# ────────────────────────────────────────────────────────────────────

popup_window = None
_gif_job     = None

def show_popup(root):
    """거북목 감지 시 경고 팝업을 표시한다. GIF 또는 빨간 배너."""
    global popup_window, _gif_job
    if popup_window is not None:
        return

    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    pw = int(sw * 0.67)
    ph = int(sh * 0.67)
    px = (sw - pw) // 2
    py = (sh - ph) // 2

    popup_window = tk.Toplevel(root)
    popup_window.geometry(f"{pw}x{ph}+{px}+{py}")
    popup_window.overrideredirect(True)
    popup_window.attributes("-topmost", True)

    try:
        gif = Image.open(IMAGE_PATH)
        frames = []
        delays = []
        try:
            while True:
                frame = gif.copy().convert("RGBA").resize((pw, ph), Image.LANCZOS)
                frames.append(ImageTk.PhotoImage(frame))
                delays.append(gif.info.get("duration", 80))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

        lbl = tk.Label(popup_window, bd=0, bg="black")
        lbl.pack()

        def animate(idx=0):
            global _gif_job
            if popup_window is None:
                return
            lbl.config(image=frames[idx])
            lbl.image = frames[idx]
            next_idx = (idx + 1) % len(frames)
            _gif_job = popup_window.after(delays[idx], animate, next_idx)

        animate()

    except Exception as e:
        print(f"이미지 로드 실패: {e}")
        popup_window.configure(bg="#CC0000")
        tk.Label(popup_window, text="Sit up straight!",
                 font=("Arial", 60, "bold"), fg="white", bg="#CC0000").pack(expand=True)

    with state_lock:
        state["popup_open"] = True

def close_popup():
    """팝업 창을 닫고 GIF 애니메이션 타이머를 취소한다."""
    global popup_window, _gif_job
    if popup_window is not None:
        if _gif_job:
            popup_window.after_cancel(_gif_job)
            _gif_job = None
        popup_window.destroy()
        popup_window = None
    with state_lock:
        state["popup_open"] = False

def poll(root):
    """tkinter 메인루프에서 200ms마다 거북목 상태를 확인해 팝업을 열거나 닫는다."""
    with state_lock:
        is_turtle  = state["is_turtle"]
        popup_open = state["popup_open"]
        running    = state["running"]

    if not running:
        root.quit()
        return

    if is_turtle and not popup_open:
        show_popup(root)
    elif not is_turtle and popup_open:
        close_popup()

    root.after(200, lambda: poll(root))

# ────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()

    root = tk.Tk()
    root.withdraw()   # 메인 창 숨기기 — 팝업만 사용

    root.after(500, lambda: poll(root))
    root.mainloop()

    with state_lock:
        state["running"] = False
    t.join(timeout=3)
    print("프로그램 종료")
