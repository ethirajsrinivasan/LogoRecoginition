import common
import numpy as np
import multiprocessing

from concurrent.futures import ProcessPoolExecutor, as_completed
import util
import skimage
import os

def gen_annot_file_line(img_fn, class_name, train_subset_class, rect):
    rect = ' '.join(map(str, rect))
    line = ' '.join([img_fn, class_name, train_subset_class, rect])
    return line


def get_bg_proposals(object_proposals, annot):
    annot_rect = util.get_annot_rect(annot)

    bg_proposals = []
    for obj_proposal in object_proposals:
        if util.iou(obj_proposal, annot_rect) <= 0.5:
            bg_proposals.append(obj_proposal)
    return bg_proposals

def gen_annot_file_lines(annot):
    lines = []

    # Get original annot line
    # print annot
    img_fn, class_name, train_subset_class = util.parse_annot(annot)
    # print(img_fn,class_name,train_subset_class)
    annot_rect = util.get_annot_rect(annot)

    #
    lines.append(
        gen_annot_file_line(img_fn, class_name, train_subset_class,
                            annot_rect))
    #
    # # Load image
    img = skimage.io.imread(os.path.join(common.TRAIN_IMAGE_DIR, img_fn))
    #
    # #
    # # # Selective search
    object_proposals = util.get_object_proposals(img)
    if len(object_proposals) == 0:
        return lines
    # #
    # # # Background proposals
    bg_proposals = get_bg_proposals(object_proposals, annot)
    if len(bg_proposals) == 0:
        return lines
    # #
    # # # Select bg proposal
    bg_proposal = bg_proposals[np.random.choice(
        np.array(bg_proposals).shape[0])]
    x1, y1, x2, y2 = bg_proposal[0], bg_proposal[
        1], bg_proposal[0] + bg_proposal[2], bg_proposal[1] + bg_proposal[3]
    lines.append(
        gen_annot_file_line(img_fn, common.CLASS_NAME[-1], train_subset_class,
                            [x1, y1, x2, y2]))

    return lines

def main():
    annot_train = np.loadtxt(common.ANNOT_FILE, dtype='a')
    # print annot_train
    # print('train_annotation: {}'.format(annot_train.shape[0]))

    # Multi processing
    results = []
    #
    n_workers = multiprocessing.cpu_count()

    #
    with ProcessPoolExecutor(n_workers) as executer, open(
            common.ANNOT_FILE_WITH_BG, 'w') as fw:
        for annot in annot_train:
            results.append(executer.submit(gen_annot_file_lines, annot))

        for result in as_completed(results):
            print('\n'.join(result.result()))
            fw.writelines('\n'.join(result.result()))
            fw.writelines('\n')

main()