import cv2
import mediapipe as mp
import pygame
import numpy as np
from typing import Optional, Tuple, List


def get_picture(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    ret, frame = cap.read()
    if ret:
        return frame
    else:
        return None


def estimate_pose(
    image: np.ndarray,
    pose: mp.solutions.pose.Pose = mp.solutions.pose.Pose(static_image_mode=False),
):
    mp_pose = mp.solutions.pose
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pose_landmarks = results.pose_landmarks
    return pose_landmarks


def landmarks_to_points(
    landmarks,
    image_shape: tuple[int, int, int],
) -> List[Tuple[int, int]]:
    h, w = image_shape[:2]
    points = []
    for lm in landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
    return points


def draw_stickman(
    screen: pygame.Surface,
    points: List[Tuple[int, int]],
) -> None:
    if points:
        # 肩幅を取得
        left_shoulder = points[11]
        right_shoulder = points[12]
        shoulder_width = np.linalg.norm(
            np.array(left_shoulder) - np.array(right_shoulder)
        )

        # 首の根本(両肩の中点)を取得
        neck_base = np.array((left_shoulder) + np.array(right_shoulder)) // 2
        neck_base = tuple(neck_base.astype(int))

        # 腰骨の両端の中点を取得
        left_hip = points[23]
        right_hip = points[24]
        hip_center = np.array((left_hip) + np.array(right_hip)) // 2
        hip_center = tuple(hip_center.astype(int))

        # 頭を描画
        head_center = points[0]
        pygame.draw.circle(
            screen, (255, 0, 0), head_center, float(shoulder_width // 2), 2
        )

        # 背骨を描画
        pygame.draw.lines(screen, (0, 255, 0), False, [neck_base, hip_center], 2)

        # 腕を描画
        left_arm_points = [neck_base] + [points[i] for i in [13, 15]]
        right_arm_points = [neck_base] + [points[i] for i in [14, 16]]
        pygame.draw.lines(screen, (0, 0, 255), False, left_arm_points, 2)
        pygame.draw.lines(screen, (0, 0, 255), False, right_arm_points, 2)


def main() -> None:
    pygame.init()

    cap = cv2.VideoCapture(0)

    picture = get_picture(cap)
    if picture is None:
        print("Failed to capture initial picture")
        return
    screen = pygame.display.set_mode((picture.shape[1], picture.shape[0]))
    pygame.display.set_caption("Stickman Pose")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)

    clock = pygame.time.Clock()  # 追加: FPS制御用

    running = True
    while running:
        try:
            picture = get_picture(cap)
            if picture is not None:
                landmarks = estimate_pose(picture, pose)
                if landmarks:
                    points = landmarks_to_points(landmarks, picture.shape)

                    h, w = picture.shape[:2]
                    bg = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
                    bg = np.rot90(np.fliplr(bg))
                    bg_surface = pygame.surfarray.make_surface(bg)

                    screen.blit(bg_surface, (0, 0))
                    draw_stickman(screen, points)
                    pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            else:
                print("Failed to capture picture")

            clock.tick(30)  # 追加: 30FPSで描画
        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

    pygame.quit()
    pose.close()
    cap.release()


if __name__ == "__main__":
    main()
