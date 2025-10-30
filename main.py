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
    pose: mp.solutions.pose.Pose,
):
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


def line_intersection(p1, p2, p3, p4):
    """2直線の交点を計算"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        # 平行な場合は首の根本を返す
        return p1

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (int(px), int(py))


def draw_stickman(
    screen: pygame.Surface,
    points: List[Tuple[int, int]],
) -> None:
    if points:
        left_shoulder = points[11]
        right_shoulder = points[12]
        shoulder_width = np.linalg.norm(
            np.array(left_shoulder) - np.array(right_shoulder)
        )

        neck_base = tuple((np.array(left_shoulder) + np.array(right_shoulder)) // 2)
        left_hip = points[23]
        right_hip = points[24]
        hip_center = tuple((np.array(left_hip) + np.array(right_hip)) // 2)
        head_center = points[0]

        line_width = int(shoulder_width // 10)
        line_color = (55, 55, 55)
        filled_color = (233, 233, 233)

        shoulder_line = (left_shoulder, right_shoulder)
        spine_line = (head_center, hip_center)
        arm_root = line_intersection(
            shoulder_line[0], shoulder_line[1], spine_line[0], spine_line[1]
        )

        # 背骨
        pygame.draw.lines(
            screen, line_color, False, [head_center, hip_center], line_width
        )
        # 両端を丸く
        pygame.draw.circle(screen, line_color, head_center, line_width // 2)
        pygame.draw.circle(screen, line_color, hip_center, line_width // 2)

        # 腕
        left_arm_points = [arm_root] + [points[i] for i in [13, 15]]
        right_arm_points = [arm_root] + [points[i] for i in [14, 16]]
        pygame.draw.lines(screen, line_color, False, left_arm_points, line_width)
        pygame.draw.lines(screen, line_color, False, right_arm_points, line_width)
        # 両端を丸く
        for point in [*left_arm_points, *right_arm_points]:
            pygame.draw.circle(screen, line_color, point, line_width // 2)

        # 頭
        pygame.draw.circle(screen, filled_color, head_center, int(shoulder_width // 2))
        pygame.draw.circle(
            screen, line_color, head_center, int(shoulder_width // 2), line_width
        )


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
                landmarks = estimate_pose(np.fliplr(picture), pose)
                if landmarks:
                    points = landmarks_to_points(landmarks, picture.shape)

                    h, w = picture.shape[:2]
                    screen.fill((0, 0, 255))

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
