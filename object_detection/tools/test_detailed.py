import argparse
import os
import warnings
import json

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
import evaluation
from tqdm import tqdm
from tabulate import tabulate

################################################################################

def csv_write(out_csv_filename,
    cls_report,
    coco_all,
    obj_all):

    with open(out_csv_filename, 'w') as csv_stat:
        csv_stat.write('|====classification summary====|\n')
        for s in cls_report:
            for val in s:        
                csv_stat.write(f'{val}')

                if val != s[-1]:
                    csv_stat.write(',')
            csv_stat.write('\n')
        csv_stat.write('\n')
        csv_stat.write('|====coco evaluation summary====|\n')

        csv_stat.write('|**summary: all**|\n')
        for s in coco_all:
            for val in s:        
                csv_stat.write(f'{val}')

                if val != s[-1]:
                    csv_stat.write(',')
            csv_stat.write('\n')
        csv_stat.write('\n')
        csv_stat.write('|**summary: object area**|\n')
        for s in obj_all:
            for val in s:        
                csv_stat.write(f'{val}')

                if val != s[-1]:
                    csv_stat.write(',')
            csv_stat.write('\n')

    csv_stat.close()
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--db', type=str, help='dataset name')
    parser.add_argument('--dbpath', default='../dataset', type=str, help='dataset directory path')
    parser.add_argument('--testimg', type=str, help='test dataset image directory path')
    parser.add_argument('--testgt', type=str, help='test dataset gt file path')
    parser.add_argument('--statpath', default='./statistics', type=str, help='dataset directory path')
    parser.add_argument('--out', help='output result file in pickle format')
    # parser.add_argument("--gtfile", 
    #     type=str, 
    #     help="gt json file path")
    parser.add_argument("--predfile", 
        type=str, 
        help="pred json file path")
    parser.add_argument("--outfile", 
        type=str, 
        help="output conf matrix file")
    parser.add_argument("--outcsv", 
        type=str, 
        help="output csv file for statistics")
    parser.add_argument("--conf_iou", 
        type=float,
        default=0.5, 
        help="confusion matrix iou threahold")
    parser.add_argument("--coco_iou", 
        type=float,
        default=0.5,
        choices=[0.95, 0.5, 0.75],
        help="coco iou threahold")
    # parser.add_argument("--ap_type", 
    #     type=str,
    #     default='all',
    #     choices=['all', 'objarea'],
    #     help="mAP all: all | Object-wise ap: objarea")
    # parser.add_argument("--areasize", 
    #     type=int,
    #     default=32,
    #     help="area for small object")
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)


    ####custom dataset
    if args.statpath:
        os.makedirs(args.statpath, exist_ok=True)

    cls_name = []

    with open(f'{args.dbpath}/{args.db}/annotation/{args.db}_train.json') as f:
       json_data = json.load(f)
    for data_id, data_info in json_data.items():
        if data_id == 'categories':
            for cat in data_info:
                cls_name.append(cat['name'])
    cls_name = tuple(cls_name)
    
    cfg.data_root = args.dbpath

    if args.testimg is not None and args.testgt is not None:
        cfg.data.test.ann_file = args.testgt
        cfg.data.test.img_prefix = args.testimg
        cfg.data.test.classes = cls_name

    else:
        cfg.data.test.ann_file = f'{args.dbpath}/{args.db}/annotation/{args.db}_test.json'
        cfg.data.test.img_prefix = f'{args.dbpath}/{args.db}/image/test/'
        cfg.data.test.classes = cls_name
        args.testgt = f'{args.dbpath}/{args.db}/annotation/{args.db}_test.json'

    # cfg.data.train.ann_file = f'{args.dbpath}/{args.db}/annotation/{args.db}_train.json'
    # cfg.data.train.img_prefix = f'{args.dbpath}/{args.db}/image/train/'
    # cfg.data.train.classes = cls_name

    # cfg.data.val.ann_file = f'{args.dbpath}/{args.db}/annotation/{args.db}_train.json'
    # cfg.data.val.img_prefix = f'{args.dbpath}/{args.db}/image/train/'
    # cfg.data.val.classes = cls_name

    # modify num classes of the model in box head
    # modify num classes of the model in box head
    if args.model_name == 'mask_rcnn' or args.model_name == 'carafe':
        cfg.model.roi_head.bbox_head.num_classes = len(cls_name)
        cfg.model.roi_head.mask_head.num_classes = len(cls_name)    

    if args.model_name == 'htc':
        cfg.model.roi_head.bbox_head[0].num_classes = len(cls_name)
        cfg.model.roi_head.bbox_head[1].num_classes = len(cls_name)
        cfg.model.roi_head.bbox_head[2].num_classes = len(cls_name)
        cfg.model.roi_head.mask_head[0].num_classes = len(cls_name)
        cfg.model.roi_head.mask_head[1].num_classes = len(cls_name)
        cfg.model.roi_head.mask_head[2].num_classes = len(cls_name)
        cfg.data.train.seg_prefix = f'{args.dbpath}/{args.db}/image/train/'
    
    if args.model_name == 'cascade_mask_rcnn':
        cfg.model.roi_head.bbox_head[0].num_classes = len(cls_name)
        cfg.model.roi_head.bbox_head[1].num_classes = len(cls_name)
        cfg.model.roi_head.bbox_head[2].num_classes = len(cls_name)
        cfg.model.roi_head.mask_head.num_classes = len(cls_name)

    if args.model_name == 'cascade_rcnn':
        cfg.model.roi_head.bbox_head[0].num_classes = len(cls_name)
        cfg.model.roi_head.bbox_head[1].num_classes = len(cls_name)
        cfg.model.roi_head.bbox_head[2].num_classes = len(cls_name)

    if args.model_name == 'yolact':
        cfg.model.bbox_head.num_classes = len(cls_name)
        cfg.model.mask_head.num_classes = len(cls_name)
        cfg.model.segm_head.num_classes = len(cls_name)   
    
    if args.model_name == 'freeanchor':
        cfg.model.bbox_head.num_classes = len(cls_name)
    
    if args.model_name == 'ssd':
        cfg.model.bbox_head.num_classes = len(cls_name)
    
    if args.model_name == 'detr':
        cfg.model.bbox_head.num_classes = len(cls_name)
    
    if args.model_name == 'deformable_detr':
        cfg.model.bbox_head.num_classes = len(cls_name)

    if args.model_name == 'fcos':
        cfg.model.bbox_head.num_classes = len(cls_name)
    
    if args.model_name == 'fsaf':
        cfg.model.bbox_head.num_classes = len(cls_name)

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    # calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'|__Total parameters: {total_params}')
    # print(f'|__Total trainable parameters: {total_trainable_params}')
    # 

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
        
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))

    ####Evaluation statistics
    print('\n\n|__Dataset evaluation\n')
    # gt_path = args.gtfile
    # pred_path = args.predfile
    # output_image = args.outfile
    # confusion_matrix_iou_threshold = args.conf_iou

    # params = evaluation.build_params(args.coco_iou, args.areasize)  # Params for COCO metrics
    # performance_evaluation = evaluation.DetectionPerformanceEvaluation(args.testgt, args.predfile, params=params,
    #                                                         th=args.conf_iou)
    # performance_evaluation.build_confussion_matrix(args.outfile,args.outcsv)
    # performance_evaluation.run_coco_metrics(args.ap_type, args.coco_iou, args.areasize, args.outcsv)

    coco_summary = []    

    ap_type = ['all', 'objarea']

    for ap in tqdm(ap_type):
        
        if ap == 'all':
            coco_iou = 0.95 #set to 0.95 for default coco evaluation
            areasize = 32 #set to 32 for default coco evaluation
            params = evaluation.build_params(coco_iou, areasize)  # Params for COCO metrics
            performance_evaluation = evaluation.DetectionPerformanceEvaluation(args.testgt, args.predfile, params=params,
                                                                    th=args.conf_iou)
            cls_report = performance_evaluation.build_confussion_matrix(args.outfile,args.outcsv)
            summary = performance_evaluation.run_coco_metrics(ap, coco_iou, areasize, args.outcsv)
            coco_summary.append(summary)
        
        if ap == 'objarea':
            areasize = [20, 32]

            for a in areasize:

                params = evaluation.build_params(args.coco_iou, a)  # Params for COCO metrics
                performance_evaluation = evaluation.DetectionPerformanceEvaluation(args.testgt, args.predfile, params=params,
                                                                        th=args.conf_iou)
                # performance_evaluation.build_confussion_matrix(output_image,args.outcsv)
                summary = performance_evaluation.run_coco_metrics(ap, args.coco_iou, a, args.outcsv)

                coco_summary.append(summary)

    print('\n\n|====classification summary====|\n')
    # print(cls_report)

    print(tabulate(cls_report, 
        headers="firstrow",
        tablefmt="psql"))

    coco_all = coco_summary[0]
    obj_all = [coco_summary[1][0],
        coco_summary[1][1],
        coco_summary[2][1],
        coco_summary[2][2],
        coco_summary[2][3]]

    print('\n\n|====coco evaluation summary====|\n')    
    print('\n|**summary: all**|\n')
    print(tabulate(coco_all, 
        headers="firstrow",
        tablefmt="psql")) 
    
    print('\n|**summary: object area**|\n')
    
    print(tabulate(obj_all, 
        headers="firstrow",
        tablefmt="psql")) 

    csv_write(args.outcsv,
        cls_report,
        coco_all,
        obj_all)

    print('\n[done]\n')
    

if __name__ == '__main__':
    main()
