import cv2
import os

def extract_frames(video_path, output_directory):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) and frame count
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {fps}")
    print(f"Total Frames: {total_frames}")

    # Loop through each frame and save as an image
    for frame_number in range(total_frames):
        ret, frame = cap.read()

        if not ret:
            print("Error reading frame.")
            break

        # Save the frame as an image
        frame_filename = os.path.join(output_directory, f"frame_{frame_number:04d}.png")
        cv2.imwrite(frame_filename, frame)

        # Print progress
        if frame_number % 100 == 0:
            print(f"Extracting frame {frame_number}/{total_frames}")

    # Release the video capture object
    cap.release()
    print("Frame extraction complete.")



if __name__ == "__main__":
    # base_dir = UCF101_subset
    base_dir = 'UCF101_subset'
    folders = os.listdir(os.path.join(base_dir))
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        for sub_folder in os.listdir(folder_path):
            sub_folder_path = os.path.join(folder_path, sub_folder)
            for video in os.listdir(sub_folder_path):
                if video.endswith('.avi'):
                    video_path = os.path.join(sub_folder_path, video)
                    output_directory = os.path.join(sub_folder_path, 'frames')
                    extract_frames(video_path, output_directory)
