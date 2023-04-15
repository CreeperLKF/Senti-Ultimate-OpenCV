import cv2
import numpy as np

import os
import sys
import getopt

DILATE_CORE = (3, 3)
DILATE_ITER = 8
GAUSS_CORE = (9, 9)
hsv_green = np.uint8([50, 248, 255])
tolerance = 10
HOUGH_DP = 1
HOUGH_MINDIST = 25
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 15
HOUGH_MINCIRCLE = 225
CIRCLE_RAD_OFFSET = 15
STABLE_ANALYZE_LEN = 5
STABLE_Y_RANGE = 25
STABLE_X_RANGE = 50
FLICKER_RANGE = 20

def replace_circle_with_image(video_path, image_path, output_video_path):
    # Read the image and split the alpha channel
    overlay_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    overlay_image_rgb = overlay_image[:, :, :3]
    overlay_image_alpha = overlay_image[:, :, 3]

    # Open the video
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Create the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    init_x, init_y, init_rate = 0, 0, 0
    out_frames = []
    out_circles = []

    # Find the first stable circle
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Detect green circles
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.uint8(np.clip(hsv_green * (1 - tolerance / 100), 0, 255))
        upper_green = np.uint8(np.clip(hsv_green * (1 + tolerance / 100), 0, 255))
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        dilate_mask = cv2.dilate(mask, DILATE_CORE, iterations=DILATE_ITER)
        out_frames.append(dilate_mask)
        
        hough_mask = cv2.GaussianBlur(mask, GAUSS_CORE, 2, 2)
        circles = cv2.HoughCircles(hough_mask, cv2.HOUGH_GRADIENT, HOUGH_DP, HOUGH_MINDIST, 
                                   param1=HOUGH_PARAM1, param2=HOUGH_PARAM2, minRadius=HOUGH_MINCIRCLE)
        # print(circles)
        if circles is None:
            circles = [[]]
        
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0]:
                x, y, r = np.int0(circle)
                r += CIRCLE_RAD_OFFSET
                if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                    continue
                r -= CIRCLE_RAD_OFFSET
                out_circles.append((x, y, r))
                break
            else:
                out_circles.append(None)

    # Analyze stable circles
    for i in range(STABLE_ANALYZE_LEN, len(out_circles) + 1):
        circles = out_circles[i-STABLE_ANALYZE_LEN:i]
        if None in circles:
            continue
        circles = np.array(circles)
        if max(circles[:,0]) - min(circles[:,0]) > STABLE_Y_RANGE or \
            max(circles[:,1]) - min(circles[:,1]) > STABLE_X_RANGE:
            continue
        vx, vy, x, y = cv2.fitLine(np.transpose([np.linspace(0, STABLE_ANALYZE_LEN, STABLE_ANALYZE_LEN), 
                                                 circles[:,2]]), cv2.DIST_L1, 0, 1e-2, 1e-2)
        cx, cy = np.mean(circles[:,0]), np.mean(circles[:,1])
        for j in range(i-STABLE_ANALYZE_LEN):
            rhat = ((j - (i - STABLE_ANALYZE_LEN) - x) * vy / vx * 1.25) + y
            out_circles[j] = (int(cx) + np.random.randint(-FLICKER_RANGE, FLICKER_RANGE), 
                              int(cy) + np.random.randint(-FLICKER_RANGE, FLICKER_RANGE), int(rhat))
            # print(out_circles[j])
        # print(circles)
        # print(i, len(out_circles))

        break

    # Replace the circle with the image
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        cnt = int(video.get(cv2.CAP_PROP_POS_FRAMES) - 1)
        dilate_mask = out_frames[cnt]
        circle = out_circles[cnt]

        if circle is not None:
            x, y, r = circle
            r += CIRCLE_RAD_OFFSET

            a, b, c, d = max(0, y-r), min(height, y+r), max(0, x-r), min(width, x+r)
            e, f, g, h = max(0, r-y), min(2*r, r+height-y), max(0, r-x), min(2*r, r+width-x)

            # Resize overlay image to match the circle size
            resized_overlay_image = cv2.resize(overlay_image_rgb, (2 * r, 2 * r))
            resized_overlay_alpha = cv2.resize(overlay_image_alpha, (2 * r, 2 * r))
            overlay_alpha = (dilate_mask[a:b, c:d] == 255) & (resized_overlay_alpha[e:f, g:h] > 0)

            # Replace the circle with the overlay image
            frame[a:b, c:d] = np.uint8(np.clip(cv2.add(
                resized_overlay_image[e:f, g:h] * overlay_alpha[..., None],
                frame[a:b, c:d] * (np.uint8(1) - overlay_alpha[..., None])
            ), 0, 255))

        # Write the output frame
        out.write(frame)

    video.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    argv = sys.argv[1:]
    input_folder = None
    output_folder = None
    image_path = "image.png"
    video_path = "video.mp4"
    output_video_path = "result.mp4"
    final_video_path = "target.mp4"
    param_prompt = "senti.py [-i <inputfile>] [-o <outputfile>] [-I <inputfolder> -O <outputfolder>]"

    try:
        opts, args = getopt.getopt(argv, "i:o:I:O:")
    except getopt.GetoptError:
        print(param_prompt)
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(param_prompt)
            sys.exit()
        elif opt == "-i":
            image_path = arg
        elif opt == "-o":
            final_video_path = arg
        elif opt == "-I":
            input_folder = arg
        elif opt == "-O":
            output_folder = arg
    
    if input_folder is not None or output_folder is not None:
        if input_folder is None or output_folder is None:
            print(param_prompt)
            sys.exit(2)
        if not os.path.isdir(input_folder):
            print("Input folder does not exist")
            sys.exit(1)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        print("processing folder {}...".format(input_folder))
        for input_path in os.listdir(input_folder):
            if input_path.endswith(".png") or input_path.endswith(".jpg") or input_path.endswith(".webp"):
                image_path = os.path.join(input_folder, input_path)
                final_video_path = os.path.join(output_folder, input_path[:input_path.rfind('.')] + ".mp4")
                if os.path.isfile(final_video_path):
                    print("passed {}".format(image_path))
                    continue
                print("processing {}...".format(image_path))
                replace_circle_with_image(video_path, image_path, output_video_path)
                os.system("ffmpeg -hide_banner -loglevel error "
                        "-y -i {} -i {} -c copy -map 0:0 -map 1:1 -shortest {}"\
                            .format(output_video_path, video_path, final_video_path))
    else:
        print("processing {}...".format(image_path))
        replace_circle_with_image(video_path, image_path, output_video_path)
        os.system("ffmpeg -hide_banner -loglevel error "
                "-y -i {} -i {} -c copy -map 0:0 -map 1:1 -shortest {}"\
                    .format(output_video_path, video_path, final_video_path))
        
    

