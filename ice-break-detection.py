import cv2
import numpy as np
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path

def detect_main_ice_break(video_path, output_dir, output_formats=None):


    # Default formats
    if output_formats is None:
        output_formats = ['mp4', 'avi', 'mov', 'wmv']

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Output name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join(output_dir, f"ice_break_detection_{timestamp}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert pixels → kilometers
    lake_width_km = 60
    pixel_to_km_ratio = lake_width_km / width

    # Colors
    colors = {
        'primary': (0, 120, 212),
        'secondary': (255, 140, 0),
        'accent': (16, 137, 62),
        'highlight': (230, 230, 0),
        'background': (45, 45, 45),
        'text': (240, 240, 240),
        'grid': (100, 100, 100),
        'alert': (220, 53, 69),
        'water': (32, 99, 155)
    }

    # Video writers
    video_writers = {}
    codec_map = {
        'mp4': 'mp4v',
        'avi': 'XVID',
        'mov': 'mp4v',
        'wmv': 'WMV2'
    }

    for fmt in output_formats:
        if fmt in codec_map:
            fourcc = cv2.VideoWriter_fourcc(*codec_map[fmt])
            output_path = f"{output_base}.{fmt}"
            video_writers[fmt] = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"✔ Output initialized: {output_path}")

    # Window
    cv2.namedWindow("Ice Break Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ice Break Detection", 640, 480)

    width_history = []
    distance_history = []
    frame_count = 0

    # Process frames
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = frame.copy()

        # Header
        cv2.rectangle(result_frame, (0, 0), (width, 60), colors['background'], -1)
        cv2.putText(result_frame, "ICE BREAK DETECTION",
                    (int(width/2) - 250, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, colors['text'], 2)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Blue detection
        lower_blue = np.array([90, 30, 40])
        upper_blue = np.array([130, 255, 255])

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = np.ones((7, 7), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            main_contour = contours[0]

            cv2.drawContours(result_frame, [main_contour], -1, colors['accent'], 4)

            leftmost = tuple(main_contour[main_contour[:, :, 0].argmin()][0])
            rightmost = tuple(main_contour[main_contour[:, :, 0].argmax()][0])

            actual_width_pixels = rightmost[0] - leftmost[0]
            estimated_width_km = actual_width_pixels * pixel_to_km_ratio

            shore_distance_km = leftmost[0] * pixel_to_km_ratio

            width_history.append(estimated_width_km)
            distance_history.append(shore_distance_km)

            cv2.putText(result_frame,
                        f"Width: {actual_width_pixels}px ({estimated_width_km:.2f} km)",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors['text'], 2)

        # Frame count display
        cv2.putText(result_frame,
                    f"Frame: {frame_count}/{total_frames}",
                    (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)

        cv2.imshow("Ice Break Detection", result_frame)

        frame_count += 1

        if cv2.waitKey(1) == ord('q'):
            break

        for writer in video_writers.values():
            writer.write(result_frame)

    # Trend plot
    if len(width_history) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(width_history, linewidth=2)
        plt.grid(True)
        plt.title("Ice Break Width Trend (km)")
        plt.xlabel("Frame")
        plt.ylabel("Width (km)")
        trend_path = f"{output_base}_trend.png"
        plt.savefig(trend_path)
        print(f"✔ Trend saved: {trend_path}")

    cap.release()
    for writer in video_writers.values():
        writer.release()
    cv2.destroyAllWindows()

    print("\n🎉 Processing Complete!")
    print("Saved Files:")
    for fmt in video_writers.keys():
        print(f" - {output_base}.{fmt}")



def main():
    # FIXED PATHS (NO ERROR)
    video_path = r"C:\Users\iTparK\Desktop\New folder\Automated-Ice-Break-Detection\ic.mp4"
    output_dir = r"C:\Users\iTparK\Desktop\New folder\Automated-Ice-Break-Detection\ice_output"

    output_formats = ['mp4', 'avi', 'mov', 'wmv']

    detect_main_ice_break(video_path, output_dir, output_formats)


if __name__ == "__main__":
    main()
