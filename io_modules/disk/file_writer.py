import cv2


def write_file(file_path, binary_content):
    with open(file_path, "wb") as f:
        f.write(binary_content)


def video_writer(image_dir, num_of_files, file_name):
    img_array = []
    for p in range(num_of_files):
        curr_file = image_dir + str(p) + ".jpg"
        img = cv2.imread(curr_file)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    out = cv2.VideoWriter(file_name + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for p in range(len(img_array)):
        out.write(img_array[p])
    out.release()
