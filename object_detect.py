# Module for detecting objects in a given image based on a model trained on
# coco (common objects in context) dataset
# Note in yolo height is iterated over first, so images should be interpreted as:
# (batch , width , height, channels)

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import colorsys
import random
import matplotlib.pyplot as plt
import cv2


# ----------------Helper functions----------------------------------


def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def draw_boxes(image, scores, boxes, classes, class_names, colors):
    """
    draws boxes around the detected objects in the image
    :param image: PIL object of the image on which object detection is being performed
    :param scores: tensor of shape (None,) containig values {pc} for all bounding boxes
    :param boxes: tensor of shape (None, 4) containing values {y1,x1,y2,x2} for all boxes
    :param classes: tensor of shape (None,) containing indices of classes identified by the bounding box
    :param class_names: a list of all the class names
    :param colors: a list of color corresponding to each class name
    """
    # getting font
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1])//300

    for i,c in reversed( list( enumerate(classes) ) ):
        predicted_class = class_names[c]
        box= boxes[i]
        score= scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        # Creating a draw object
        draw = ImageDraw.Draw(image)
        # getiing label_size in pixels
        label_size = draw.textsize(label, font)  # of numpy array form

        y1, x1, y2, x2 = box
        x1 = max(0, np.floor(x1 + 0.5).astype('int32'))   # 0.5 for rounding off
        y1 = max(0, np.floor(y1+0.5).astype('int32'))
        x2 = min(image.size[0], np.floor(x2 + 0.5).astype('int32'))
        y2 = min(image.size[1], np.floor(y2 + 0.5).astype('int32'))

        # setting origin for label to be printed
        if y1 - label_size[1] >= 0:
            text_origin = np.array([x1, y1-label_size[1]])
        else:
            text_origin = np.array([x1, y1])

        # drawing the boxes and labels
        for i in range(thickness):
            draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline=colors[c])

        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0,0,0), font=font)
        del draw


def generate_colors(class_names):
    """
    generates a list of colors for a list of class_names
    :param class_names: list of different class names of objects to be identified
    :return: a list of colors , each for every class name
    """
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[1] * 255)), colors))
    random.seed(1000)
    random.shuffle(colors)  # to remove similarity in colors of adjacent classes
    random.seed(None)  # reset seed to default

    return colors


def preprocess_image(img_path, model_input_img_size):
    """
    Prepares the image to be input into the yolo model
    :param img_path: the path of the image file
    :param model_input_img_size: the input size of image that the model accepts(608,608) for our model
                                 <type> : tuple
    :return:
    image : The PIL object of our image file
    image_data : the numpy array containing the normalized pixel values of our resized image to be input
                 into the model
    """

    image = Image.open(img_path)
    resized_img = image.resize(model_input_img_size, Image.BICUBIC)
    img_data = np.array(resized_img, dtype='float32')
    img_data /= 255  # normalizing data
    img_data = np.expand_dims(img_data, 0)  # adding batch dimension

    return image, img_data


def read_anchors(anchor_path):
    """
    reads anchor s from path specified and returns the width and height of anchors in numpy form
    :param anchor_path: path to anchor file containing width and height of each anchor box
    :return: a numpy array containing width and height of each anchor box of shape :
             (number of anchors, 2)
    """
    with open(anchor_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    return anchors


def process_features(features, anchors, num_classes=80):
    """
    processes the output of yolo model for detecting bounding boxes in appropriate format
    :param features: the output yolo model of shape (None, 19, 19, 425)
    :param anchors: tensor of shape (num of anchors, 2) containing width and height of each anchor
    :param num_classes: total number of different classes
    :returns:
    1.) box_confidence : tensor of shape (?,19, 19, 5, 1) containing {pc}
    2.) box_xy : tensor of shape (?, 19, 19, 5, 2) containing (bx, by)
    3.) box_wh : tensor of shape (?, 19, 19, 5, 2) containing (bw, bh)
    4.) box_class_probs : tensor of shape (?, 19, 19, 5, 80) containing {c1,c2,c3....,c80}
    """
    num_anchors = len(anchors)
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])  # (batch, width ,height, anchors, 2)

    conv_dims = K.shape(features)[1:3]  # width and height divisions of image
    conv_height_divs = conv_dims[1]
    conv_width_divs = conv_dims[0]
    conv_height_index = K.arange(0, stop=conv_height_divs)
    conv_height_index = K.tile(conv_height_index, [conv_width_divs])
    conv_width_divs = K.arange(0, stop=conv_width_divs)
    conv_width_index = K.tile(K.expand_dims(conv_width_divs, 0), [conv_height_divs, 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(features))

    feats = K.reshape(features, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(features))

    box_confidence = K.sigmoid(feats[..., 4:5])
    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_class_probs = K.softmax(feats[..., 5:])

    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_confidence, box_xy, box_wh, box_class_probs


def boxes_to_corners(boxes_xy, boxes_wh):
    """
    converts yolo outputs boxes_xy and boxes _wh into corners for all the bounding boxes
    :param boxes_xy: tensor of shape (19, 19, 5, 2) of values {bx, by} representing the mid points of boxes
    :param boxes_wh: tensor of shape (19, 19, 5, 2) of values {bw, bh} representing the width and height
                     of the bounding boxes
    :return:
    boxes: tensor of shape (19, 19, 5, 4) of values {y1, x1, y2, x2} where (x1,y1) [upper corner]
           and (x2,y2) [lower corner] are the corners of the bounding boxes
    """
    boxes_x1y1 = boxes_xy - (boxes_wh / 2.)
    boxes_x2y2 = boxes_xy + (boxes_wh / 2.)

    boxes = K.concatenate([
        boxes_x1y1[..., 1:2],
        boxes_x1y1[..., 0:1],
        boxes_x2y2[..., 1:2],
        boxes_x2y2[..., 0:1]
    ])  # indexing done this way to preserve the shape otherwise last axis is collapsed

    return boxes


def scale_boxes(boxes, input_shape):
    """
    scales boxes generated for image of size (608, 608, 3) into boxes for image with size specifies by
    input_shape
    :param boxes: tensor of shape (19, 19, 5, 4) of values {y1, x1, y2, x2} for each bounding box
    :param input_shape: tuple containing shape of image in float (height, width)
    :return:
    boxes - scaled to input_shape
    """
    height = input_shape[0]
    width = input_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    return boxes


def filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
    """
    Filters out the bounding box predictions from yolo model and outputs boxes with class probibilities
    greater than the threshold specified(default = 0.6)
    ---Note---
    Each bounding box is of the shape (19,19,5,85)
    where we have 5 anchor boxes and each anchor box has labels {pc, bx, by, bw, bh, c1 ,c2 ,c3, ....., c80}
    Total classes :80
    :param box_confidence: A tensor of shape (19, 19 ,5 ,1) - containing the value pc for each anchor box
    :param boxes: Tensor of shape(19 ,19 ,5 ,4) - containing values {by1, bx1, by2, bx2}
                  where (x1.y1) and (x2,y2) are the two corners of the bounding box
    :param box_class_probs: Tensor of shape (19, 19, 5, 80) - containing values {c1, c2, c3, ..., c80}
    :param threshold: for selecting bounding boxes with class prob > threshold
    :return:
    1.) scores - tensor of shape (None,) containing the class probability score for the boxes filtered
    2.) boxes - tensor of shape (None, 4) containing the values {by1, bx1, by2, bx2} for each bounding box
    3.) classes - tensor of shape (None,) containing the index of class detected by each bounding box
    """
    box_scores = box_confidence * box_class_probs  # calculating prob for each class
    box_class_indexes = K.argmax(box_scores, axis=-1)
    box_max_scores = K.max(box_scores, axis=-1)

    filtering_mask = box_max_scores > threshold  # tensor of values True or false

    # keeping class indexes for boxes with max score > threshold
    classes = tf.boolean_mask(tensor=box_class_indexes, mask=filtering_mask)
    # keeping max scores for boxes with max score > threshold
    scores = tf.boolean_mask(tensor=box_max_scores, mask=filtering_mask)
    # keeping box dimensions for boxes with max score > threshold
    boxes = tf.boolean_mask(tensor=boxes, mask=filtering_mask)

    return scores, boxes, classes


def non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Implements non max suppression over given bounding boxes
    :param scores: tensor (None,) of class scores for each bounding box
    :param boxes: tensor (None, 4) of dimensions for each bounding box
    :param classes: tensor (None,) of indexes of class detected by each bounding box
    :param max_boxes: integer type scalar for maximum number of boxes to output after non max suppression
    :param iou_threshold: threshold for performing non max suppression (nms)
    :return:
    1.) scores - tensor of shape (None,) containing the class probability score for the boxes after nms
    2.) boxes - tensor of shape (None, 4) containing the values {by1, bx1, by2, bx2} for each bounding box
    3.) classes - tensor of shape (None,) containing the index of class detected by each bounding box
    """

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initializing the tensor variable

    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold,
                                               name='non_max_suppression')

    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes


def get_boxes(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=0.6,
              iou_threshold=0.5):
    """
    combines the functions filter_boxes and non_max_suppression to get the final bounding boxes to be displayed
    scaled according to the shape of the input_image
    :param yolo_outputs: a tuple that contains the outputs of our yolo model and has following elemenst :
                         box_confidence - tensor of shape (19, 19, 5, 1) - contains - {pc}
                         boxes_xy - shape - (19, 19, 5, 2)-contains- {bx, by}(midpoints)
                         boxes_wh - shape - (19, 19, 5, 2)-contains- {bw, bh}
                         box_class_probs - shape- (19, 19, 5, 80) -contains- {c1, c2, c3,..., c80}
    :param image_shape: a tuple containing dimensions of input image (height, width) -float type
    :param max_boxes: max boxes to be displayed
    :param score_threshold: threshold for detecting class of object identified by each bounding box
    :param iou_threshold: threshold for non_max_suppression
    :return:
    1.) scores - tensor of shape (None,) containing the class probability score for the boxes after nms
    2.) boxes - tensor of shape (None, 4) containing the values {by1, bx1, by2, bx2} for each bounding box
    3.) classes - tensor of shape (None,) containing the index of class detected by each bounding box
    """

    box_confidence, boxes_xy, boxes_wh, box_class_probs = yolo_outputs
    boxes = boxes_to_corners(boxes_xy, boxes_wh)
    scores, boxes, classes = filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
    scores, boxes, classes = non_max_suppression(scores, boxes, classes,
                                                 max_boxes, iou_threshold)
    boxes = scale_boxes(boxes, image_shape)

    return scores, boxes, classes


def predict(sess, img_path):
    """
    Runs the graph on image on which object detection is to be performed
    :param sess: Keras session object
    :param img_path: path to the image on which object detection is to be performed
    """

    # Preprocess image
    img, img_data = preprocess_image(img_path, model_input_img_size=(608,608))

    # Run the graph to compute scores, boxes, classes
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: img_data,
                                                             K.learning_phase():0}) # 0 for testing

    # generate colors
    colors= generate_colors(class_names)

    # draw bounding boxes
    draw_boxes(img, out_scores, out_boxes, out_classes, class_names, colors)

    # display results
    plt.imshow(img)

    # save the image
    # img.save('path where is to be saved', quality=90)

    return out_scores, out_boxes, out_classes


def predict_cam(sess, frame, model_input_img_size=(608,608)):
    """
    Runs the graph on image on which object detection is to be performed
    :param sess: Keras session object
    :param frame: numpy image data
    """

    # Preprocess image
    img = Image.fromarray(frame)
    resized_img = img.resize(model_input_img_size, Image.BICUBIC)
    img_data = np.array(resized_img, dtype='float32')
    img_data /= 255  # normalizing data
    img_data = np.expand_dims(img_data, 0)  # adding batch dimension

    # Run the graph to compute scores, boxes, classes
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: img_data,
                                                             K.learning_phase():0}) # 0 for testing

    # generate colors
    colors= generate_colors(class_names)

    # draw bounding boxes
    draw_boxes(img, out_scores, out_boxes, out_classes, class_names, colors)

    # display results
    img_data = np.array(img)
    cv2.imshow('yolo output',img_data)

    # save the image
    # img.save('path where is to be saved', quality=90)

    return out_scores, out_boxes, out_classes


# ------------------------Create a session-------------------


sess = K.get_session()


# ----------------------------------------------------------------

anchors = read_anchors('YOLO_model/model_anchors.txt')

class_names = read_classes('YOLO_model/coco_classes.txt')

# ---------------- Creating computation graph -------------------

yolo_model = load_model('YOLO_model/model.h5')
# yolo_model.summary()

yolo_outputs = process_features(yolo_model.output, anchors)

image_shape= (720., 1280.)  # (height,width)

scores, boxes, classes = get_boxes(yolo_outputs, image_shape)

# --------------------------Run the graph on image/cam stream----------------------------------------

"""  For cam:

url='https://192.168.43.212:8080'+'/video' # url that IP webcam shows
vs = cv2.VideoCapture(url)


i=0
while True:
    ret, frame= vs.read()
    if not ret:
        continue
    if i%5==0:
        out_scores, out_boxes, out_classes = predict_cam(sess, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i+=1

cv2.destroyAllWndows()
"""

# for single picture:
out_scores, out_boxes, out_classes = predict(sess, 'YOLO_model/test.jpg')