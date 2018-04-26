import sys
import tensorflow as tf
import numpy as np
from scipy.misc import imresize
import common
from model import cnn as model
import argparse
import os
import preprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import util
import skimage.draw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_fn', help='image filename')
    return parser.parse_args()

def logo_recognition(sess, img, obj_proposal, graph_params):
    recog_results = {}
    recog_results['obj_proposal'] = obj_proposal

    if img.shape != common.CNN_SHAPE:
        img = imresize(img, common.CNN_SHAPE, interp='bicubic')
    img = preprocess.scaling(img)
    img = img.reshape((1, common.CNN_IN_HEIGHT, common.CNN_IN_WIDTH,
                       common.CNN_IN_CH)).astype(np.float32)
    pred = sess.run(
        [graph_params['pred']], feed_dict={graph_params['target_image']: img})
    recog_results['pred_class'] = common.CLASS_NAME[np.argmax(pred)]
    recog_results['pred_prob'] = np.max(pred)
    return recog_results


def setup_graph():
    graph_params = {}
    graph_params['graph'] = tf.Graph()
    with graph_params['graph'].as_default():
        model_params = model.params()
        graph_params['target_image'] = tf.placeholder(
            tf.float32,
            shape=(1, common.CNN_IN_HEIGHT, common.CNN_IN_WIDTH,
                   common.CNN_IN_CH))
        logits = model.cnn(
            graph_params['target_image'], model_params, keep_prob=1.0)
        graph_params['pred'] = tf.nn.softmax(logits)
        graph_params['saver'] = tf.train.Saver()
    return graph_params

def detect_logo(target_image):
    import time
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    import cv2
    if len(target_image.shape) >= 2 and target_image.shape[2] == 4:
        target_image=cv2.cvtColor(target_image, cv2.COLOR_BGRA2BGR)
    # skimage.io.imsave("/home/viki/Desktop/DataWarehousing/project/test_target.jpg",target_image)
    # brand=None
    # target_image=util.load_target_image("/home/viki/Desktop/DataWarehousing/project/test_target.jpg")
    object_proposals = util.get_object_proposals(target_image)
    graph_params = setup_graph()
    sess = tf.Session(graph=graph_params['graph'])
    tf.global_variables_initializer()
    if os.path.exists('models'):
        save_path = os.path.join('models', 'deep_logo_model')
        graph_params['saver'].restore(sess, save_path)
        # print('Model restored')
    # else:
        # print('Initialized')

    results = []
    i=0
    for obj_proposal in object_proposals:
        i+=1
        x, y, w, h = obj_proposal
        crop_image = target_image[y:y + h, x:x + w]
        results.append(
            logo_recognition(sess, crop_image, obj_proposal, graph_params))
    del_idx = []
    for i, result in enumerate(results):
        if result['pred_class'] == common.CLASS_NAME[-1]:
            del_idx.append(i)
    results = np.delete(results, del_idx)
    nms_results = util.nms(results, pred_prob_th=0.999999, iou_th=0.4)
    fig =  plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(target_image)
    if len(nms_results)>0:
        for result in nms_results:
            (x, y, w, h) = result['obj_proposal']
            brand=result['pred_class']
    #         ax.text(
    #             x,
    #             y,
    #             result['pred_class'],
    #             fontsize=13,
    #             bbox=dict(facecolor='red', alpha=0.7))
    #         rect = mpatches.Rectangle(
    #             (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    #         ax.add_patch(rect)
    # fig.savefig('/home/viki/brand_emotion/static/img/%s.jpg'%brand)

    # print "------------------------------"
    # print brand
    return brand