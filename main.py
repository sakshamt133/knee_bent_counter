import cv2 as cv
import mediapipe as mp
import numpy as np
import time

start = 'down'
reps = 0


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    return np.abs(radians*180.0/np.pi)


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv.VideoCapture('KneeBendVideo.mp4')
is_open = True
temp = 0

result = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc(*'MJPG'), 10, (int(cap.get(3)), int(cap.get(4))))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while is_open:
        is_open, frame = cap.read()
        width = cap.get(3)
        height = cap.get(4)

        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img.flags.writeable = False

        results = pose.process(img)

        img.flags.writeable = True
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        cv.rectangle(img, (0, 0), (img.shape[1], 75), (0, 225, 225), -1)

        try:
            landmarks = results.pose_landmarks.landmark
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            angle = calculate_angle(hip, knee, ankle)

            cv.putText(img, str(angle), tuple(np.multiply(knee, [width, height]).astype(int)), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 255), 2, cv.LINE_AA)

            if angle < 140 and start == 'down':
                start = 'up'
                temp = 0
                up_time = time.time()
                curr_time = up_time

            if angle >= 140 and start == 'up' and temp >= 8:
                start = 'down'
                reps += 1

            if angle > 140 and start == 'up' and temp < 8:
                cv.putText(img, "Keep your knee bent", (350, 50), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (255, 255, 255), 1, cv.LINE_AA)

            if time.time() >= (curr_time + 1):
                temp += 1
                curr_time = time.time()

            cv.putText(img, "Time", (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

            if start == 'down':
                temp = 0

            cv.putText(img, str(temp), (80, 60), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 2,
                       (255, 255, 255), 2, cv.LINE_AA)

            cv.putText(img, "REPS", (180, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                       2, cv.LINE_AA)

            cv.putText(img, str(reps), (270, 60), cv.FONT_HERSHEY_SIMPLEX, 2,
                       (255, 255, 255), 2, cv.LINE_AA)

        except:
            pass

        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        result.write(img)
        cv.imshow("frame", img)
        if cv.waitKey(10) & 0xff == ord('q'):
            break
