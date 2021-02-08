import time
import os
import argparse
import string

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument(
    '--invid', default='', type=str,
    help='The input video.')
parser.add_argument(
    '--mask', default='', type=str,
    help='The mask video.')
parser.add_argument(
    '--outname', default='', type=str,
    help='The output video.')
parser.add_argument(
    '--codec', default='DIVX', type=str,
    help='The used encoder.')
parser.add_argument(
    '--checkpoint_dir', default='', type=str,
    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    #ng.get_gpus(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] =''
    args = parser.parse_args()

    sess_config = tf.compat.v1.ConfigProto()
    tf.compat.v1.disable_eager_execution()
    sess_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=sess_config)

    model = InpaintCAModel()
    input_image_ph = tf.compat.v1.placeholder(
        tf.float32, shape=(1, 680, 680*2, 3))
    output = model.build_server_graph(FLAGS, input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.train.load_variable(
            args.checkpoint_dir, from_name)
        assign_ops.append(tf.compat.v1.assign(var, var_value))
    sess.run(assign_ops)

    print('Model loaded.')


    inpuVid = cv2.VideoCapture(args.invid)
    maskVid = cv2.VideoCapture(args.mask)
    outpVid = None#cv2.VideoWriter(args.outname, cv2.VideoWriter_fourcc(*args.codec), 30.0,(2720,1530))
    #if not outpVid.isOpened():
    #    exit()
    i = 0

    t = time.time()

    while(True):
        # Capture frame-by-frame
        ret, inFrame = inpuVid.read()
        if not ret:
            break
        ret, maFrame = maskVid.read()
        if not ret:
            break

        if outpVid == None:
            outpVid = cv2.VideoWriter(args.outname, cv2.VideoWriter_fourcc(*args.codec), 30.0,(int(inFrame.shape[1]),int(inFrame.shape[0])))
            if not outpVid.isOpened():
                exit()

        # Find contours
        imgray = cv2.cvtColor(maFrame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        subims = []
        for c in contours:
            xmin, ymin, w, h = cv2.boundingRect(c)
            xmax = xmin+w
            ymax = ymin+h
            found = False
            for si in subims:
                sixMin, siyMin, sixMax, siyMax = si
                if xmin < sixMin:
                    sixMin = xmin
                if ymin < siyMin:
                    siyMin = ymin
                if xmax > sixMax:
                    sixMax = xmax
                if ymax > siyMax:
                    siyMax = ymax
                if sixMax - sixMin < 500 and siyMax - siyMin < 500:
                    found = True
                    subims.remove(si)
                    subims.append((sixMin, siyMin, sixMax, siyMax))
                    break
            if not found:
                subims.append((xmin, ymin, xmax, ymax))
        
        outImg = inFrame.copy()
        j = 0
        for si in subims:
            sixMin, siyMin, sixMax, siyMax = si
            sixMin = min(max(sixMin - 90, 0), inFrame.shape[1]-1-680)
            siyMin = min(max(siyMin - 90, 0), inFrame.shape[0]-1-680)
            sixMax = sixMin + 680
            siyMax = siyMin + 680
            image = inFrame[siyMin:siyMax, sixMin:sixMax]
            mask = maFrame[siyMin:siyMax, sixMin:sixMax]

            h, w, _ = image.shape
            grid = 4
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid, :]
            print('Shape of image: {}'.format(image.shape))

            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)

            # current_graph = tf.compat.v1.get_default_graph()

            # all_names = [op.name for op in current_graph.get_operations()]
            # print("\n".join(all_names))
            
            # load pretrained model
            result = sess.run(output, feed_dict={input_image_ph: input_image})
            print('Processed: {} {}'.format(i, j))
            outImg[siyMin:siyMax, sixMin:sixMax] = result[0][:, :, ::-1]
            j += 1
        outpVid.write(outImg.astype(np.uint8)[0:inFrame.shape[0],0:inFrame.shape[1]])
        i += 1
    outpVid.release()
    inpuVid.release()
    maskVid.release()
    print('Time total: {}'.format(time.time() - t))
