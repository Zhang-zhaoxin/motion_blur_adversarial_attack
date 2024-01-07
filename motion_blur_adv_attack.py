import os
import colorsys
import random
import cv2

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import matplotlib.pyplot as plt
graph = tf.get_default_graph()

def letterbox_image_1(image, w, h, nw, nh):
    """
    resize image with unchanged aspect ratio using padding
    图像截取
    """
    iw, ih = image.size

    # if iw > ih:
    #     cbox = [(w-nw)//2, (h-nh)//2, w, (h-nh)//2 + nh]
    # else:
    cbox = [(w-nw)//2, (h-nh)//2, (w-nw)//2 + nw, (h-nh)//2 + nh]

    image_cropped = image.crop(cbox)

    return image_cropped
    
def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    # TODO: move this function to utils
    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    # loss is averaged over the batch so need to multiply the batch
    # size to find the actual gradient of each input sample

    assert isinstance(p, float) or isinstance(p, int)
    norm = np.linalg.norm(x)
    norm = np.maximum(norm, 1 * small_constant)
    return (1. / norm) * x

class Yolo4(object):
    def get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_yolo(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.class_names = self.get_class()
        self.anchors = self.get_anchors()

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        self.sess = K.get_session()

        # Load model, or construct model and load weights.
        self.yolo4_model = yolo4_body(Input(shape=(608, 608, 3)), num_anchors // 3, num_classes)
        self.yolo4_model.load_weights(model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        if self.gpu_num >= 2:
            self.yolo4_model = multi_gpu_model(self.yolo4_model, gpus=self.gpu_num)

        self.input_image_shape = K.placeholder(shape=(2,))
        self.boxes, self.scores, self.classes = yolo_eval(self.yolo4_model.output, self.anchors,
                                                          len(self.class_names), self.input_image_shape,
                                                          score_threshold=self.score)

    def __init__(self, score, iou, anchors_path, classes_path, model_path, gpu_num=1):
        self.score = score
        self.iou = iou
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.model_path = model_path
        self.gpu_num = gpu_num
        self.load_yolo()

    def close_session(self):
        self.sess.close()

    def detect_image(self, image, model_image_size=(608, 608)):
        start = timer()

        boxed_image, w, h, nw, nh, iw, ih = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo4_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print(out_boxes, out_scores, out_classes)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            print(out_classes)
            #exit()
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        return image


    def Attack(self, image, count, jpgfile, model_image_size=(608, 608)):
        # sess = K.get_session()
        global graph
        boxed_image, w, h, nw, nh, iw, ih = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        original_image = np.copy(image_data)
        cost_function = tf.reduce_sum(self.scores)
        print("cost_function:{}".format(cost_function))
        gradient_function = K.gradients(cost_function, self.yolo4_model.input)[0]
        cost = 1
        e = 5 / 255
        n = 0
        index = 0
        # 最大改变幅度
        max_change_above = original_image + 0.1
        max_change_below = original_image - 0.1
        # 主要攻击循环
        while cost > 0.002:
        # for i in range(0, 10):
            img = image_data[0]
            img *= 255. 
            im = Image.fromarray(img.astype(np.uint8))
            im = letterbox_image_1(im, w, h, nw, nh)
            im = im.resize((iw, ih), Image.BICUBIC) # 填充图像
                            
            im.save(os.path.join("output/blur_adv_image", os.path.basename(jpgfile)))
            # 再次打开图片
            im = Image.open(os.path.join("output/blur_adv_image", os.path.basename(jpgfile)))
            
            im, w, h, nw, nh, iw, ih = letterbox_image(im, tuple(reversed(model_image_size)))

            image_data = np.array(im, dtype='float32')
            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)
            # 计算梯度
            # with graph.as_default():
            cost, gradients, out_classes, out_boxes = self.sess.run(
                [cost_function, gradient_function, self.classes, self.boxes],
                feed_dict={
                    self.yolo4_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
            print("batch:{} Cost: {:.8}".format(index, cost))

            # 计算对抗噪声
            n = np.sign(gradients)
            image_data -= n * e
            image_data = np.clip(image_data, max_change_below, max_change_above)
            image_data = np.clip(image_data, 0, 1.0)

            index += 1
            if cost < 0.002:
                break
        return 0

if __name__ == '__main__':
    model_path = 'yolo4_weight.h5'
    anchors_path = 'model_data/yolo4_anchors.txt'
    classes_path = 'model_data/coco_classes.txt'


    score = 0.5
    iou = 0.5
    model_image_size = (608, 608)

    import glob
    count = 0
    path = "output/blur_images/*.jpg"
    for jpgfile in glob.glob(path):
        yolo4_model = Yolo4(score, iou, anchors_path, classes_path, model_path)
        img = Image.open(jpgfile)
        result = yolo4_model.Attack(img, count, jpgfile, model_image_size=model_image_size)
        count += 1
        print(count)
        K.clear_session()  # 将动态图进行清除，否则节点增多增大内存消耗并且程序报错
    yolo4_model.close_session()


