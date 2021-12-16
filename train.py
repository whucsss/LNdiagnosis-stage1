import os
from PIL import Image
import keras
import numpy as np
import random
import datetime
import tensorflow as tf
from utils import visualize
from utils.config import Config
from utils.anchors import get_anchors
from utils.utils import mold_inputs, unmold_detections
from nets.mrcnn import get_train_model, get_predict_model
from nets.mrcnn_training import data_generator, load_image_gt
from dataset import ShapesDataset


def log(text, array=None):
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(), array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("", ""))
        text += "  {}".format(array.dtype)
    print(text)


class ShapesConfig(Config):
    NAME = "shapes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # BATCH_SIZE = 1
    NUM_CLASSES = 1 + 5
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512


if __name__ == "__main__":
    datetime = datetime.datetime.now().strftime('%Y%m%d')
    learning_rate = 1e-4
    init_epoch = 0
    epoch = 100

    dataset_root_path = "./train_dataset/"
    img_floder = dataset_root_path + "imgs/"
    mask_floder = dataset_root_path + "mask/"
    yaml_floder = dataset_root_path + "yaml/"
    imglist = os.listdir(img_floder)

    count = len(imglist)
    np.random.seed(10101)
    np.random.shuffle(imglist)
    print(imglist)

    train_imglist = imglist[:int(count * 0.66)]
    val_imglist = imglist[int(count * 0.66):]

    MODEL_DIR = "logs"

    COCO_MODEL_PATH = "model_data/mask_rcnn_coco.h5"
    config = ShapesConfig()

    config.STEPS_PER_EPOCH = len(train_imglist) // config.IMAGES_PER_GPU
    config.VALIDATION_STEPS = len(val_imglist) // config.IMAGES_PER_GPU
    config.display()

    # 训练数据集准备
    dataset_train = ShapesDataset()
    dataset_train.load_shapes(len(train_imglist), img_floder, mask_floder, train_imglist, yaml_floder)
    dataset_train.prepare()

    # 验证数据集准备
    dataset_val = ShapesDataset()
    dataset_val.load_shapes(len(val_imglist), img_floder, mask_floder, val_imglist, yaml_floder)
    dataset_val.prepare()

    # 获得训练模型
    model = get_train_model(config)
    model.summary()
    model.load_weights(COCO_MODEL_PATH, by_name=True, skip_mismatch=True)

    # 数据生成器
    train_generator = data_generator(dataset_train, config, shuffle=True, batch_size=config.BATCH_SIZE)
    val_generator = data_generator(dataset_val, config, shuffle=True, batch_size=config.BATCH_SIZE)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    callbacks = [
        keras.callbacks.TensorBoard(log_dir=MODEL_DIR,
                                    histogram_freq=0, write_graph=True, write_images=False),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR,  datetime + "_epoch{epoch:03d}_loss{loss:.3f}_val_loss{val_loss:.3f}.h5"),
            verbose=0, save_weights_only=True, save_best_only=True),
    ]

    if True:
        log("\nStarting at epoch {}. LR={}\n".format(init_epoch, learning_rate))
        log("Checkpoint Path: {}".format(MODEL_DIR))

        # optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1= 0.9,beta_2=0.999,
        #                                   clipnorm=config.GRADIENT_CLIP_NORM)
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=config.LEARNING_MOMENTUM,
            clipnorm=config.GRADIENT_CLIP_NORM)
        # optimizer = keras.optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=config.WEIGHT_DECAY)

        # 设置一下loss信息
        model._losses = []
        model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss", "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = model.get_layer(name)
            if layer.output in model.losses:
                continue
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * config.LOSS_WEIGHTS.get(name, 1.))
            model.add_loss(loss)

        model.compile(
            optimizer=optimizer,
            loss=[None] * len(model.outputs)
        )

        for name in loss_names:
            if name in model.metrics_names:
                print(name)
                continue
            layer = model.get_layer(name)
            model.metrics_names.append(name)
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * config.LOSS_WEIGHTS.get(name, 1.))
            model.metrics_tensors.append(loss)

        model.fit_generator(
            train_generator,
            initial_epoch=init_epoch,
            epochs=epoch,
            steps_per_epoch=config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=config.VALIDATION_STEPS,
        )
