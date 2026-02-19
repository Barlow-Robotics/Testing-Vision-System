import cv2
import numpy as np
import mss
class Camera:
    def __init__(self, camera_index=1):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0) # change exposure
        self.mode = 'camera'  # or 'screen'

    def run(self):
        while True:
            if self.mode == 'camera':
                while True:
                    ret, frame = self.cap.read()
            
                    if not ret:
                        print("Error: Could not read frame")
                        break

                    self.compileImage(frame)
            elif self.mode == 'screen':
                with mss.mss() as sct:
                    monitor = sct.monitors[1]  # 1 = primary screen
                    while True:
                        screenshot = sct.grab(monitor)
                        frame = np.array(screenshot)
                        self.compileImage(frame)


    def end(self):
        self.cap.release()
        cv2.destroyAllWindows()


    def compileImage(self, frame):
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # v = cv2.equalizeHist(v)
        hsv = cv2.merge((h, s, v))

        lower_yellow = np.array([18, 120, 100])
        upper_yellow = np.array([32, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0
        area_ratio = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 2000:
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            # if circularity < :
            #     continue

            (x, y), r = cv2.minEnclosingCircle(c)
            if r < 20:
                continue

            expected_area = np.pi * r * r
            area_ratio = area / expected_area
            if area_ratio < 0.4 or area_ratio > 0.95:
                continue
            circle_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.circle(circle_mask, (int(x), int(y)), int(r), 255, -1)

            # Count yellow pixels inside the circle
            yellow_inside = cv2.bitwise_and(mask, mask, mask=circle_mask)

            yellow_pixels = cv2.countNonZero(yellow_inside)
            circle_pixels = cv2.countNonZero(circle_mask)

            if circle_pixels == 0:
                continue

            yellow_ratio = yellow_pixels / circle_pixels
            if yellow_ratio < 0.6:
                continue
            score = circularity * area  # quality metric
            if score > best_score:
                best_score = score
                best = (int(x), int(y), int(r), circularity, area)
        if best:
            x, y, r, circ, area = best
            cv2.circle(frame, (x,y), r, (0,255,0), 2)
            cv2.circle(frame, (x,y), 4, (0,0,255), -1)
            print(f"Ball at {(x,y)} r={r} circ={circ:.2f} area={area:.1f}")
            print(f"Detected yellow object at {area_ratio} with radius {r} with circularity {circularity:.2f} with perimeter {perimeter:.2f} with area {area:.2f}")

        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)

if __name__ == "__main__":
    camera = Camera()
    camera.run()
    # camera.end()