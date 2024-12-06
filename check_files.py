import cv2
import os

def remove_unreadable_images(root_folder):
    """
    Removes unreadable image files and non-image files from all subdirectories in the root folder.
    Args:
        root_folder (str): Path to the root folder containing image subdirectories.
    """
    unreadable_files = []
    non_image_files = []
    total_images_checked = 0

    for subdir, _, files in os.walk(root_folder):
        for file_name in files:
            file_path = os.path.join(subdir, file_name)

            # Check if the file is an image
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                total_images_checked += 1  # Increment the counter
                try:
                    # Try reading the image
                    image = cv2.imread(file_path)
                    if image is None:
                        print(f"Unreadable image found and removed: {file_path}")
                        unreadable_files.append(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    unreadable_files.append(file_path)
            else:
                # If the file is not an image, mark it for removal
                print(f"Non-image file found and removed: {file_path}")
                non_image_files.append(file_path)

    # Remove unreadable image files
    for file_path in unreadable_files:
        os.remove(file_path)

    # Remove non-image files
    for file_path in non_image_files:
        os.remove(file_path)

    print(f"Checked {total_images_checked} images in total.")
    print(f"Removed {len(unreadable_files)} unreadable images.")
    print(f"Removed {len(non_image_files)} non-image files.")
