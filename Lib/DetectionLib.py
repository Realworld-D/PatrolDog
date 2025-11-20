import os
import yaml
import glob
from functools import reduce

import cv2
import numpy as np
import math
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
from random import randint
import sys
# add deploy path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'])))
sys.path.insert(0, parent_path)

from preprocess import Resize, NormalizeImage, Permute, PadStride, LetterBoxResize, WarpAffine

# Global dictionary
SUPPORT_MODELS = {
    'YOLO', 'RCNN', 'SSD', 'Face', 'FCOS', 'SOLOv2', 'TTFNet', 'S2ANet', 'JDE',
    'FairMOT', 'DeepSORT', 'GFL', 'PicoDet', 'CenterNet', 'TOOD',
    'StrongBaseline', 'STGCN'
}

class PaddleDetector(object):
    """
    Args:
        pred_config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
        enable_mkldnn_bfloat16 (bool): whether to turn on mkldnn bfloat16
        output_dir (str): The path of output
        threshold (float): The threshold of score for visualization
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT. 
                                    Used by action model.
    """

    def __init__(self,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 enable_mkldnn_bfloat16=False,
                 output_dir='output',
                 threshold=0.5,
                 delete_shuffle_pass=False):
        self.pred_config = self.set_config(model_dir)
        self.predictor, self.config = load_predictor(
            model_dir,
            run_mode=run_mode,
            batch_size=batch_size,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            device=device,
            use_dynamic_shape=self.pred_config.use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            enable_mkldnn_bfloat16=enable_mkldnn_bfloat16,
            delete_shuffle_pass=delete_shuffle_pass)
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.threshold = threshold
        self.color_list = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(len(self.pred_config.labels))]

    def set_config(self, model_dir):
        return PredictConfig(model_dir)

    def preprocess(self, img):
        im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_info = {
            'scale_factor': np.array(
                [1., 1.], dtype=np.float32),
            'im_shape': None,
        }
        
        im_info['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
        im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
        
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))
        for operator in preprocess_ops:
            # print(operator)
            im, im_info = operator(im, im_info)
        
        input_im_lst = []
        input_im_info_lst = []
        input_im_lst.append(im)
        input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        return inputs

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        np_boxes_num = result['boxes_num']
        if np_boxes_num[0] <= 0:
            # print('[WARNNING] No object detected.')
            result = {'boxes': np.zeros([0, 6]), 'boxes_num': [0]}
            result = {k: v for k, v in result.items() if v is not None}
            return result

        result = {k: v for k, v in result.items() if v is not None}

        # label:0     monkey    threshold>=0.8
        # label:1     panda     threshold>=0.95
        # label:2     wolf      threshold>=0.9
        # result = dict(boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
        np_boxes = result['boxes']
        new_np_boxes = []
        for idx in range(np_boxes.shape[0]):
            cls_id = int(np_boxes[idx, 0])
            # bbox = np_boxes[idx, 2:]
            score = np_boxes[idx, 1]

            if cls_id == 0 and score >= 0.8:
                new_np_boxes.append(np_boxes[idx])
            if cls_id == 1 and score >= 0.95:
                new_np_boxes.append(np_boxes[idx])
            if cls_id == 2 and score >= 0.9:
                new_np_boxes.append(np_boxes[idx])

        if new_np_boxes == []:
            new_np_boxes.append([0, 0, 0, 0, 0, 0])

        new_result = {'boxes': np.array(new_np_boxes)}

        return new_result

    def filter_box(self, result, threshold):
        np_boxes_num = result['boxes_num']
        boxes = result['boxes']
        start_idx = 0
        filter_boxes = []
        filter_num = []
        for i in range(len(np_boxes_num)):
            boxes_num = np_boxes_num[i]
            boxes_i = boxes[start_idx:start_idx + boxes_num, :]
            idx = boxes_i[:, 1] > threshold
            filter_boxes_i = boxes_i[idx, :]
            filter_boxes.append(filter_boxes_i)
            filter_num.append(filter_boxes_i.shape[0])
            start_idx += boxes_num
        boxes = np.concatenate(filter_boxes)
        filter_num = np.array(filter_num)
        filter_res = {'boxes': boxes, 'boxes_num': filter_num}
        return filter_res

    def predict(self):
        '''
        Args:
            repeats (int): repeats number for prediction
        Returns:
            result (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's result include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        # model prediction
        np_boxes, np_masks = None, None
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        boxes_tensor = self.predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        boxes_num = self.predictor.get_output_handle(output_names[1])
        np_boxes_num = boxes_num.copy_to_cpu()
        if self.pred_config.mask:
            masks_tensor = self.predictor.get_output_handle(output_names[2])
            np_masks = masks_tensor.copy_to_cpu()
        result = dict(boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
        return result

    def merge_batch_result(self, batch_result):
        if len(batch_result) == 1:
            return batch_result[0]
        res_key = batch_result[0].keys()
        results = {k: [] for k in res_key}
        for res in batch_result:
            for k, v in res.items():
                results[k].append(v)
        for k, v in results.items():
            results[k] = np.concatenate(v)
        return results

    def predict_image(self,
                      image,
                      visual=False):
                   
        # preprocess
        inputs = self.preprocess(image)
        result = []
        # model prediction
        result = self.predict()

        # # postprocess
        result = self.postprocess(inputs, result)
        
        return result

    def visualize_boxes(self, img, result, threshold=0.5):
        np_boxes = result['boxes']
        expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
        np_boxes = np_boxes[expect_boxes, :]
        # for dt in np_boxes:
        #     clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
        #     print(clsid, detector.pred_config.labels[clsid], score)
        # clsid = np_boxes[:, 0].astype(int)
        # bbox = np_boxes[:, 2:]
        # score = np_boxes[:, 1]
        for idx in range(np_boxes.shape[0]):
            cls_id = int(np_boxes[idx, 0])
            bbox = np_boxes[idx, 2:]
            score = np_boxes[idx, 1]
            
            color = self.color_list[cls_id]
            xmin, ymin, xmax, ymax = bbox
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

            label = self.pred_config.labels[cls_id]
            cv2.rectangle(img, (int(xmin), int(ymin-15)), (int(xmin+80), int(ymin)), color, -1)
            cv2.putText(img, f"{label} {str(round(score,2))}", (int(xmin), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        
        return img
    
    
def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        imgs (list(numpy)): list of images (np.ndarray)
        im_info (list(dict)): list of image info
    Returns:
        inputs (dict): input of model
    """
    inputs = {}

    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs['image'] = np.array((imgs[0], )).astype('float32')
        inputs['im_shape'] = np.array(
            (im_info[0]['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info[0]['scale_factor'], )).astype('float32')
        return inputs

    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

    inputs['im_shape'] = np.concatenate(im_shape, axis=0)
    inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs['image'] = np.stack(padding_imgs, axis=0)
    return inputs


class PredictConfig():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        if 'mask' in yml_conf:
            self.mask = yml_conf['mask']
        self.tracker = None
        if 'tracker' in yml_conf:
            self.tracker = yml_conf['tracker']
        if 'NMS' in yml_conf:
            self.nms = yml_conf['NMS']
        if 'fpn_stride' in yml_conf:
            self.fpn_stride = yml_conf['fpn_stride']
        if self.arch == 'RCNN' and yml_conf.get('export_onnx', False):
            print(
                'The RCNN export model is used for ONNX and it only supports batch_size = 1'
            )
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type 
        """
        for support_model in SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
            'arch'], SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


def load_predictor(model_dir,
                   run_mode='paddle',
                   batch_size=1,
                   device='CPU',
                   min_subgraph_size=3,
                   use_dynamic_shape=False,
                   trt_min_shape=1,
                   trt_max_shape=1280,
                   trt_opt_shape=640,
                   trt_calib_mode=False,
                   cpu_threads=1,
                   enable_mkldnn=False,
                   enable_mkldnn_bfloat16=False,
                   delete_shuffle_pass=False):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16/trt_int8)
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT. 
                                    Used by action model.
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need device == 'GPU'.
    """
    if device != 'GPU' and run_mode != 'paddle':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect device=='GPU', but device == {}"
            .format(run_mode, device))
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams'))
    if device == 'GPU':
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    elif device == 'XPU':
        config.enable_lite_engine()
        config.enable_xpu(10 * 1024 * 1024)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        if enable_mkldnn:
            try:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
                if enable_mkldnn_bfloat16:
                    config.enable_mkldnn_bfloat16()
            except Exception as e:
                print(
                    "The current environment does not support `mkldnn`, so disable mkldnn."
                )
                pass

    precision_map = {
        'trt_int8': Config.Precision.Int8,
        'trt_fp32': Config.Precision.Float32,
        'trt_fp16': Config.Precision.Half
    }
    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=1 << 25,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[run_mode],
            use_static=False,
            use_calib_mode=trt_calib_mode)

        if use_dynamic_shape:
            min_input_shape = {
                'image': [batch_size, 3, trt_min_shape, trt_min_shape]
            }
            max_input_shape = {
                'image': [batch_size, 3, trt_max_shape, trt_max_shape]
            }
            opt_input_shape = {
                'image': [batch_size, 3, trt_opt_shape, trt_opt_shape]
            }
            config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
                                              opt_input_shape)
            print('trt set dynamic shape done!')

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    if delete_shuffle_pass:
        config.delete_pass("shuffle_channel_detect_pass")
    predictor = create_predictor(config)
    return predictor, config


model_dir = "./model/picodet_l_320_coco_lcnet"
# file_name = "./monkey.jpg"
file_name = "./cap_wolf.jpg"
file_dir = "./image_test"
threshold = 0.5

if __name__ == "__main__":
    detector = PaddleDetector(model_dir, device='CPU')
    # for file in os.listdir(file_dir):
    #     file_name = os.path.split(file)[-1]
    #     img = cv2.imread(os.path.join(file_dir, file_name))
    # # print(img)
    #     result = detector.predict_image(img, visual=True)
    
    img = cv2.imread(file_name)
    # print(img)
    result = detector.predict_image(img)
    # print(result)
    result_img = detector.visualize_boxes(img, result)
    cv2.imwrite("./result.jpg", result_img)
    # np_boxes = result['boxes']
    # expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    # np_boxes = np_boxes[expect_boxes, :]
    # for dt in np_boxes:
    #     clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
    #     print(clsid, detector.pred_config.labels[clsid], score)