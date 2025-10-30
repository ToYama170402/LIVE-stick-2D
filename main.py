import cv2
import mediapipe as mp


def get_picture():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        return None


def estimate_pose(image):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pose_landmarks = results.pose_landmarks
    pose.close()
    return pose_landmarks


def draw_landmarks(image, landmarks):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    if landmarks:
        mp_drawing.draw_landmarks(image, landmarks, mp_pose.POSE_CONNECTIONS)
    return image


def main():
    picture = get_picture()
    if picture is not None:
        landmarks = estimate_pose(picture)
        picture_with_landmarks = draw_landmarks(picture, landmarks)
        cv2.imshow("Pose Estimation", picture_with_landmarks)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to capture picture")


if __name__ == "__main__":
    main()
