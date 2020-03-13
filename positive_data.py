import os
from PIL import Image
import numpy as np
import utils
import traceback

anno_src = r"D:\image_50000\label_50000.txt"
img_dir = r"D:\image_50000\image"

save_path = r"D:\face"


for face_size in [96]:

    print("gen %i image" % face_size)
    # 样本图片存储路径
    positive_image_dir = os.path.join(save_path, str(face_size), "positive")

    for dir_path in [positive_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 样本描述存储路径
    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0

    try:
        positive_anno_file = open(positive_anno_filename, "w")

        for i, line in enumerate(open(anno_src)):
            if i < 2:
                continue
            try:
                strs = line.split()
                image_filename = strs[0].strip()
                print(image_filename)
                image_file = os.path.join(img_dir, image_filename)

                with Image.open(image_file) as img:
                    img_w, img_h = img.size
                    x = float(strs[1].strip())
                    y = float(strs[2].strip())
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())

                    x1 = float(x)
                    y1 = float(y)
                    x2 = float(x + w)
                    y2 = float(y + h)

                    px1 = float(strs[5].strip())
                    py1 = float(strs[6].strip())
                    px2 = float(strs[7].strip())
                    py2 = float(strs[8].strip())
                    px3 = float(strs[9].strip())
                    py3 = float(strs[10].strip())
                    px4 = float(strs[11].strip())
                    py4 = float(strs[12].strip())
                    px5 = float(strs[13].strip())
                    py5 = float(strs[14].strip())

                    if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue

                    boxes = [[x1, y1, x2, y2]]

                    # 计算出人脸中心点位置
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    # 使正样本和部分样本数量翻倍
                    for _ in range(7):
                        # 让人脸中心点有少许的偏移
                        w_ = np.random.randint(-w * 0.2, w * 0.2)
                        h_ = np.random.randint(-h * 0.2, h * 0.2)
                        cx_ = cx + w_
                        cy_ = cy + h_

                        # 让人脸形成正方形，并且让坐标也有少许的偏离
                        side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(1 * max(w, h)))
                        x1_ = np.max(cx_ - side_len / 2, 0)
                        y1_ = np.max(cy_ - side_len / 2, 0)
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        crop_box = np.array([x1_, y1_, x2_, y2_])

                        # 计算坐标的偏移值
                        offset_x1 = (x1 - x1_) / side_len
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        offset_px1 = (px1 - x1_) / side_len
                        offset_py1 = (py1 - y1_) / side_len
                        offset_px2 = (px2 - x1_) / side_len
                        offset_py2 = (py2 - y1_) / side_len
                        offset_px3 = (px3 - x1_) / side_len
                        offset_py3 = (py3 - y1_) / side_len
                        offset_px4 = (px4 - x1_) / side_len
                        offset_py4 = (py4 - y1_) / side_len
                        offset_px5 = (px5 - x1_) / side_len
                        offset_py5 = (py5 - y1_) / side_len

                        # 剪切下图片，并进行大小缩放
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size))

                        iou = utils.iou(crop_box, np.array(boxes))[0]
                        if iou >= 0.70:  # 正样本
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    positive_count, 1, offset_x1, offset_y1,
                                    offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                    offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                            positive_anno_file.flush()
                            face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                            positive_count += 1
            except Exception as e:
                traceback.print_exc()


    finally:
        positive_anno_file.close()