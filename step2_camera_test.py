import cv2

# 카메라 열기 (0 = 기본 내장 카메라)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("카메라 연결 성공! 'q'를 누르면 종료됩니다.")

while True:
    ret, frame = cap.read()  # 프레임 읽기

    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 좌우 반전 (거울 모드)
    frame = cv2.flip(frame, 1)

    # 화면에 텍스트 표시
    cv2.putText(frame, "Camera OK - Press Q to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.imshow("카메라 테스트", frame)  # 영상 출력

    if cv2.waitKey(1) & 0xFF == ord('q'):  # q 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
print("종료되었습니다.")
