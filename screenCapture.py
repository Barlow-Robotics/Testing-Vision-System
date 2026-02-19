import mss
import cv2
import numpy as np

with mss.mss() as sct:
    monitor = sct.monitors[1]  # 1 = primary screen
    while True:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([18, 100, 70]) # optimized numbers based on testing
        upper_yellow = np.array([32, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)        
            if area > 2000:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                if radius > 10:
                    center = (int(x), int(y))
                    radius = int(radius)
                    
                    cv2.circle(frame, center, radius, (0, 255, 0), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    
                    cv2.putText(frame, "Yellow Ball", (center[0] - 50, center[1] - radius - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Original', frame)
        cv2.imshow('Yellow Mask', mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
