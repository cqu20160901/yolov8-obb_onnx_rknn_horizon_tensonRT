import logging
from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.utils.tool_utils import init_root_logger
import math
import cv2
import numpy as np


CLASSES = ['plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court',
           'ground track field', 'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter', 'roundabout',
           'soccer ball field', 'swimming pool']

meshgrid = []

class_num = len(CLASSES)
head_num = 3
strides = [8, 16, 32]
map_size = [[80, 80], [40, 40], [20, 20]]
reg_num = 16
nms_thresh = 0.5
object_thresh = 0.25

input_height = 640
input_width = 640



class CSXYWHR:
    def __init__(self, classId, score, x, y, w, h, angle):
        self.classId = classId
        self.score = score
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle


class DetectBox:
    def __init__(self, classId, score, pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y, angle):
        self.classId = classId
        self.score = score
        self.pt1x = pt1x
        self.pt1y = pt1y
        self.pt2x = pt2x
        self.pt2y = pt2y
        self.pt3x = pt3x
        self.pt3y = pt3y
        self.pt4x = pt4x
        self.pt4y = pt4y
        self.angle = angle


def GenerateMeshgrid():
    for index in range(head_num):
        for i in range(map_size[index][0]):
            for j in range(map_size[index][1]):
                meshgrid.append(j + 0.5)
                meshgrid.append(i + 0.5)


def get_covariance_matrix(boxes):
    a, b, c = boxes.w, boxes.h, boxes.angle
    cos = math.cos(c)
    sin = math.sin(c)
    cos2 = math.pow(cos, 2)
    sin2 = math.pow(sin, 2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(obb1, obb2, eps=1e-7):
    x1, y1 = obb1.x, obb1.y
    x2, y2 = obb2.x, obb2.y
    a1, b1, c1 = get_covariance_matrix(obb1)
    a2, b2, c2 = get_covariance_matrix(obb2)

    t1 = (((a1 + a2) * math.pow((y1 - y2), 2) + (b1 + b2) * math.pow((x1 - x2), 2)) / ((a1 + a2) * (b1 + b2) - math.pow((c1 + c2), 2) + eps)) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - math.pow((c1 + c2), 2) + eps)) * 0.5

    temp1 = (a1 * b1 - math.pow(c1, 2)) if (a1 * b1 - math.pow(c1, 2)) > 0 else 0
    temp2 = (a2 * b2 - math.pow(c2, 2)) if (a2 * b2 - math.pow(c2, 2)) > 0 else 0
    t3 = math.log((((a1 + a2) * (b1 + b2) - math.pow((c1 + c2), 2)) / (4 * math.sqrt((temp1 * temp2)) + eps)+ eps)) * 0.5

    if (t1 + t2 + t3) > 100:
        bd = 100
    elif (t1 + t2 + t3) < eps:
        bd = eps
    else:
        bd = t1 + t2 + t3
    hd = math.sqrt((1.0 - math.exp(-bd) + eps))
    return 1 - hd


def nms_rotated(boxes, nms_thresh):
    pred_boxes = []
    sort_boxes = sorted(boxes, key=lambda x: x.score, reverse=True)
    for i in range(len(sort_boxes)):
        if sort_boxes[i].classId != -1:
            pred_boxes.append(sort_boxes[i])
            for j in range(i + 1, len(sort_boxes), 1):
                ious = probiou(sort_boxes[i], sort_boxes[j])
                if ious > nms_thresh:
                    sort_boxes[j].classId = -1
    return pred_boxes


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def xywhr2xyxyxyxy(x, y, w, h, angle):
    cos_value = math.cos(angle)
    sin_value = math.sin(angle)

    vec1x= w / 2 * cos_value
    vec1y = w / 2 * sin_value
    vec2x = -h / 2 * sin_value
    vec2y = h / 2 * cos_value

    pt1x = x + vec1x + vec2x
    pt1y = y + vec1y + vec2y

    pt2x = x + vec1x - vec2x
    pt2y = y + vec1y - vec2y

    pt3x = x - vec1x - vec2x
    pt3y = y - vec1y - vec2y

    pt4x = x - vec1x + vec2x
    pt4y = y - vec1y + vec2y
    return pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y


def postprocess(out):
    print('postprocess ... ')

    detect_result = []
    output = []
    for i in range(len(out)):
        output.append(out[i].reshape((-1)))

    gridIndex = -2
    cls_index = 0
    cls_max = 0

    for index in range(head_num):
        reg = output[index * 2 + 0]
        cls = output[index * 2 + 1]
        ang = output[head_num * 2 + index]

        for h in range(map_size[index][0]):
            for w in range(map_size[index][1]):
                gridIndex += 2

                if 1 == class_num:
                    cls_max = sigmoid(cls[0 * map_size[index][0] * map_size[index][1] + h * map_size[index][1] + w])
                    cls_index = 0
                else:
                    for cl in range(class_num):
                        cls_val = cls[cl * map_size[index][0] * map_size[index][1] + h * map_size[index][1] + w]
                        if 0 == cl:
                            cls_max = cls_val
                            cls_index = cl
                        else:
                            if cls_val > cls_max:
                                cls_max = cls_val
                                cls_index = cl
                    cls_max = sigmoid(cls_max)

                if cls_max > object_thresh:
                    regdfl = []
                    for lc in range(4):
                        sfsum = 0
                        locval = 0
                        for df in range(reg_num):
                            temp = math.exp(reg[((lc * reg_num) + df) * map_size[index][0] * map_size[index][1] + h * map_size[index][1] + w])
                            reg[((lc * reg_num) + df) * map_size[index][0] * map_size[index][1] + h * map_size[index][1] + w] = temp
                            sfsum += temp

                        for df in range(reg_num):
                            sfval = reg[((lc * reg_num) + df) * map_size[index][0] * map_size[index][1] + h * map_size[index][1] + w] / sfsum
                            locval += sfval * df
                        regdfl.append(locval)

                    angle = (sigmoid(ang[h * map_size[index][1] + w]) - 0.25) * math.pi

                    left, top, right, bottom = regdfl[0], regdfl[1], regdfl[2], regdfl[3]
                    cos, sin = math.cos(angle), math.sin(angle)
                    fx = (right - left) / 2
                    fy = (bottom - top) / 2

                    cx = ((fx * cos - fy * sin) + meshgrid[gridIndex + 0]) * strides[index]
                    cy = ((fx * sin + fy * cos) + meshgrid[gridIndex + 1])* strides[index]
                    cw = (left + right) * strides[index]
                    ch = (top + bottom) * strides[index]

                    box = CSXYWHR(cls_index, cls_max, cx, cy, cw, ch, angle)

                    detect_result.append(box)
    # NMS
    print('before nms num is:', len(detect_result))
    pred_boxes = nms_rotated(detect_result, nms_thresh)

    print('after nms num is:', len(pred_boxes))

    resutl = []
    for i in range(len(pred_boxes)):
        classid = pred_boxes[i].classId
        score = pred_boxes[i].score
        cx = pred_boxes[i].x
        cy = pred_boxes[i].y
        cw = pred_boxes[i].w
        ch = pred_boxes[i].h
        angle = pred_boxes[i].angle

        bw_ = cw if cw > ch else ch
        bh_ = ch if cw > ch else cw
        bt = angle % math.pi if cw > ch else (angle + math.pi / 2) % math.pi

        pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y = xywhr2xyxyxyxy(cx, cy, bw_, bh_, bt)

        bbox = DetectBox(classid, score, pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y, angle)
        resutl.append(bbox)

    return resutl


def preprocess(src_image, input_width, input_height):
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(src_image, (input_height, input_width))
    return img


def inference(model_path, image_path, input_layout, input_offset):
    # init_root_logger("inference.log", console_level=logging.INFO, file_level=logging.DEBUG)

    sess = HB_ONNXRuntime(model_file=model_path)
    sess.set_dim_param(0, 0, '?')

    if input_layout is None:
        logging.warning(f"input_layout not provided. Using {sess.layout[0]}")
        input_layout = sess.layout[0]

    origin_image = cv2.imread(image_path)
    image_h, image_w = origin_image.shape[:2]
    image = preprocess(origin_image, input_width, input_height)
    image_data = np.expand_dims(image, axis=0)

    input_name = sess.input_names[0]
    output_name = sess.output_names
    output = sess.run(output_name, {input_name: image_data}, input_offset=input_offset)

    print('inference finished, output len is:', len(output))

    out = []
    for i in range(len(output)):
        out.append(output[i])

    pred_boxes = postprocess(out)

    print('obj num is :', len(pred_boxes))

    for i in range(len(pred_boxes)):
        classId = pred_boxes[i].classId
        score = pred_boxes[i].score
        pt1x = int(pred_boxes[i].pt1x / input_width * image_w)
        pt1y = int(pred_boxes[i].pt1y / input_width * image_h)
        pt2x = int(pred_boxes[i].pt2x / input_width * image_w)
        pt2y = int(pred_boxes[i].pt2y / input_width * image_h)
        pt3x = int(pred_boxes[i].pt3x / input_width * image_w)
        pt3y = int(pred_boxes[i].pt3y / input_width * image_h)
        pt4x = int(pred_boxes[i].pt4x / input_width * image_w)
        pt4y = int(pred_boxes[i].pt4y / input_width * image_h)
        angle = pred_boxes[i].angle

        cv2.line(origin_image, (pt1x, pt1y), (pt2x, pt2y), (0, 255, 0), 2)
        cv2.line(origin_image, (pt2x, pt2y), (pt3x, pt3y), (0, 255, 0), 2)
        cv2.line(origin_image, (pt3x, pt3y), (pt4x, pt4y), (0, 255, 0), 2)
        cv2.line(origin_image, (pt4x, pt4y), (pt1x, pt1y), (0, 255, 0), 2)

        title = CLASSES[classId] + "%.2f" % score
        cv2.putText(origin_image, title, (pt1x, pt1y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('./test_onnx_result.jpg', origin_image)


if __name__ == '__main__':
    print('This main ... ')
    GenerateMeshgrid()

    model_path = './model_output/yolov8obb_quantized_model.onnx'
    image_path = './test.jpg'
    input_layout = 'NHWC'
    input_offset = 128

    inference(model_path, image_path, input_layout, input_offset)

