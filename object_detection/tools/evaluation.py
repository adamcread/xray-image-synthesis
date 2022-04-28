################################################################################
# Example : Perform live inference on image, video, webcam input
# using different object tracking algorithms.
# Copyright (c) 2021 - Yona Falinie / Neelanjan Bhowmik / Toby Breckon 
# Durham University, UK
# License : 
################################################################################

import json
import time
from typing import Union, List
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from matplotlib import pyplot as plt
from tabulate import tabulate
import argparse
import io
from contextlib import redirect_stdout
from tqdm import tqdm
################################################################################

def csv_write(out_csv_filename,
    cls_report,
    coco_all,
    obj_all):

    with open(out_csv_filename, 'w') as csv_stat:
        csv_stat.write('|====classification summary====|\n')
        for s in cls_report:
            for i, val in enumerate(s):        
                csv_stat.write(f'{val}')

                # if val != s[-1]:
                if i != len(s)-1:
                    csv_stat.write(',')
            csv_stat.write('\n')
        csv_stat.write('\n')
        csv_stat.write('|====coco evaluation summary====|\n')

        csv_stat.write('|**summary: all**|\n')
        for i,s in enumerate(coco_all):
            for val in s:        
                csv_stat.write(f'{val}')

                # if val != s[-1]:
                if i != len(s)-1:
                    csv_stat.write(',')
            csv_stat.write('\n')
        csv_stat.write('\n')
        csv_stat.write('|**summary: object area**|\n')
        for s in obj_all:
            for i, val in enumerate(s):        
                csv_stat.write(f'{val}')
               
                # if val != s[-1]:
                if i != len(s)-1:
                    csv_stat.write(',')
            csv_stat.write('\n')

    csv_stat.close()
################################################################################

class DetectionPerformanceEvaluation:

    def __init__(self, gt: Union[str, COCO], prediction: Union[List, str], params=None, th=0.5):
        if isinstance(gt, str):
            ff = io.StringIO()
            with redirect_stdout(ff):
                gt = COCO(gt)
        
        prediction_coco = dict()
        if isinstance(prediction, str):
            # print('loading detectron output annotations into memory...')
            tic = time.time()
            # ff = io.StringIO()
            # with redirect_stdout(ff):
            prediction = json.load(open(prediction, 'r'))  # Loading the json file as an array of dicts
            assert type(prediction) == list, 'annotation file format {} not supported'.format(
                type(prediction))
            # print('Done (t={:0.2f}s)'.format(time.time() - tic))

        for i, p in enumerate(prediction):
            p['id'] = i
            # p['segmentation'] = p['segmentation']
            # p['segmentation'] = []
            p['area'] = p['bbox'][2] * p['bbox'][3]
        # Adding these lines I give the detection file the xray format
        prediction_coco["annotations"] = prediction
        prediction_coco["images"] = gt.dataset["images"]
        prediction_coco["categories"] = gt.dataset["categories"]

        # COCO object instantiation
        ff = io.StringIO()
        with redirect_stdout(ff):
            prediction = COCO()
            prediction.dataset = prediction_coco
            prediction.createIndex()

            self.ground_truth = gt
            self.prediction = prediction
            self.eval = COCOeval(gt, prediction, iouType='bbox')
            self.params = self.eval.params
            self._imgIds = gt.getImgIds()
            self._catIds = gt.getCatIds()
        # catname = [cat['name'] for cat in _coco.loadCats(_coco.getCatIds())]
        self.th = th
        if params:
            ff = io.StringIO()
            with redirect_stdout(ff):
                self.params = params
                self.eval.params = params
                self.eval.params.imgIds = sorted(self._imgIds)
                self.eval.params.catIds = sorted(self._catIds)

    def _build_no_cat_params(self):
        params = Params(iouType='bbox')
        params.maxDets = [500]
        params.areaRng = [[0 ** 2, 1e5 ** 2]]
        params.areaRngLbl = ['all']
        params.useCats = 0
        params.iouThrs = [self.th]
        return params

    def build_confussion_matrix(self, 
        out_image_filename=None,
        out_csv_filename=None):
        ff = io.StringIO()
        with redirect_stdout(ff):
            params = self._build_no_cat_params()
            self.eval.params = params
            self.eval.params.imgIds = sorted(self._imgIds)
            self.eval.params.catIds = sorted(self._catIds)
            self.eval.evaluate()

        ann_true = []
        ann_pred = []

        for evalImg, ((k, _), ious) in zip(self.eval.evalImgs, self.eval.ious.items()):
            # print('--->', evalImg['gtIds'])
            # print(ann_pred)
            ann_true += evalImg['gtIds']
            if len(ious) > 0:
                valid_ious = (ious >= self.th) * ious
                matches = valid_ious.argmax(0)
                matches[valid_ious.max(0) == 0] = -1
                ann_pred += [evalImg['dtIds'][match] if match > -1 else -1 for match in matches]
            else:
                ann_pred += ([-1] * len(evalImg['gtIds']))
            # print('----> process', ann_true[-1])

        y_true = [ann['category_id'] for ann in self.ground_truth.loadAnns(ann_true)]
        y_pred = [-1 if ann == -1 else self.prediction.loadAnns(ann)[0]['category_id'] for ann in ann_pred]
        y_true = [y + 1 for y in y_true]
        y_pred = [y + 1 for y in y_pred]
                
        cats = ['background'] + [cat['name'] for _, cat in self.ground_truth.cats.items()]
        cnf_mtx = confusion_matrix(y_true, y_pred, normalize='true')
        # print(cnf_mtx)

        '''
        ####TPR/FPR/TNR
        FP = cnf_mtx.sum(axis=0) - np.diag(cnf_mtx)  
        FN = cnf_mtx.sum(axis=1) - np.diag(cnf_mtx)
        TP = np.diag(cnf_mtx)
        TN = cnf_mtx.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        PPV_array = np.array(PPV)
        PPV_array = np.nan_to_num(PPV_array)
        PPV = list(PPV_array)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)
        # f1 score
        F1 = 2.0 * (PPV * TPR) / (PPV + TPR)
        F1_array = np.array(F1)
        F1_array = np.nan_to_num(F1_array)
        F1 = list(F1_array)

        #Avg
        TPRav = round(sum(TPR[1:])/(len(TPR)-1), 3)
        TNRav = round(sum(TNR[1:])/(len(TNR)-1), 3)
        PPVav = round(sum(PPV[1:])/(len(PPV)-1), 3)
        FPRav = round(sum(FPR[1:])/(len(FPR)-1), 3)
        FNRav = round(sum(FNR[1:])/(len(FNR)-1), 3)
        ACCav = round(sum(ACC[1:])/(len(ACC)-1), 3)
        F1av  = round(sum(F1[1:])/(len(F1)-1), 3)

        stats = []
        tex_stat = f'{TPRav} & {FPRav} & {F1av} & {PPVav} & {ACCav}'
        stats.append(['TPR', str(TPRav)])
        stats.append(['FPR', str(FPRav)])
        stats.append(['F1', str(F1av)])
        stats.append(['Precesion', str(PPVav)])
        stats.append(['Accuracy', str(ACCav)])
        # stats.append(['Tex_stat', tex_stat])
        
        # print('*' * 20)
        # print('Stats from confusion Matrix:')
        # print('Class-wise:')
        # print(f'TPR: {TPR}\nTNR: {TNR}\nFPR: {FPR}\nFNR: {FNR}\nACC: {ACC}')
        # print('Overall:')
        # print(tabulate(stats, tablefmt="psql"))
        # # print(f'TPR: {TPRav}\nFPR: {FPRav}\nF Score: {F1av}\nPrecesion: {PPVav} \nAccuracy: {ACCav}')
        # # print('\nTex version:', tex_stat)
        # print('*' * 20)   
        ####
        '''
       
        cnf_mtx_display = ConfusionMatrixDisplay(confusion_matrix=cnf_mtx, 
            display_labels=cats)
        
        _, ax = plt.subplots(figsize=(10, 9))
        plt.rcParams.update({'font.size': 18})
        cnf_mtx_display.plot(ax=ax, values_format='.3f',xticks_rotation=45, cmap="cividis")
        if out_image_filename is not None:
            cnf_mtx_display.figure_.savefig(out_image_filename)
        # print(classification_report(y_true, y_pred, target_names=cats, zero_division=1))

        cls_report = classification_report(y_true, y_pred, 
            target_names=cats,output_dict=True,
            zero_division=1)
        # print(cls_report)    

        ####format classification report
        cls_report_format = []
        no_print = ['background', 'macro avg', 'weighted avg', 'accuracy']
        cls_report_format.append(['class','precision','recall','f1-score'])
        for key, val in cls_report.items():
            if key not in no_print:
                # to_write = f'{key},{str(round(val["precision"],2))},{str(round(val["recall"],2))},{str(round(val["f1-score"],2))}'
                cls_report_format.append([f'{key}',
                    str(round(val["precision"],2)),
                    str(round(val["recall"],2)),
                    str(round(val["f1-score"],2))
                ])
            if key == 'accuracy':
                # to_write = f'{key},{round(val,2)},,'
                cls_report_format.append([f'{key}',
                    round(val,2),
                    '',
                    ''
                ])

        # no_print = ['background', 'macro avg', 'weighted avg', 'accuracy']   
        # with open(out_csv_filename, 'w') as csv_stat:
        #     csv_stat.write(f'class,precision,recall,f1-score\n')
        #     for key, val in cls_report.items():
        #         if key not in no_print:
        #             to_write = key+','+ str(round(val['precision'],2))+','+str(round(val['recall'],2))+','+str(round(val['f1-score'],2))
        #             csv_stat.write(f'{to_write}\n')
        #         if key == 'accuracy':
        #             csv_stat.write(f'{key},{round(val,2)}\n')
        # csv_stat.close()
        return cls_report_format
        pass

    def run_coco_metrics(self, ap_type='all', 
        coco_iou= 0.95,
        areasize=32,
        out_csv_filename=None):

        # print('\n\n|====coco evaluation summary====|\n')
        ff = io.StringIO()
        with redirect_stdout(ff):
            self.eval.params = self.params
            self.eval.params.imgIds = sorted(self._imgIds)
            self.eval.params.catIds = sorted(self._catIds)
            self.eval.evaluate()
            self.eval.accumulate()
            # self.eval.summarize()

        f = io.StringIO()
        with redirect_stdout(f):
            self.eval.summarize()
            
        out = f.getvalue()
        # print('\n',out)
        out = out.split('\n')
        if ap_type == 'all':
            map_95 = out[0].split('=')[-1]
            map_50 = out[1].split('=')[-1]
            map_75 = out[2].split('=')[-1]

            # with open(out_csv_filename, 'a') as csv_stat:
            #     csv_stat.write(f'\n\ncoco evaluation summary\n')
            #     csv_stat.write(f'{out[0]}\n{out[1]}')
            # csv_stat.close()

        if ap_type == 'objarea':
            if areasize == 20:
                map_tiny = out[3].split('=')[-1]    
            else:
                map_small = out[3].split('=')[-1]
            map_med = out[4].split('=')[-1]
            map_large = out[5].split('=')[-1]


        cat_name = [cat['name'] for _, cat in self.ground_truth.cats.items()]
        
        ap_50 = []
        ap_95 = []
        ap_75 = []
        obj_tiny = []
        obj_small = []
        obj_med = []
        obj_large = []
        summary = []

        # print('\n\n|__Class-wise coco evaluation')
        for c_id in self._catIds:
            # print(f'\n|____Category: {c_id} : {cat_name[c_id-1]}')
            ff = io.StringIO()
            with redirect_stdout(ff):
                self.eval.params.catIds = [c_id]
                self.eval.evaluate()
                self.eval.accumulate()
                # self.eval.summarize()
            f = io.StringIO()
            with redirect_stdout(f):
                self.eval.summarize()
            out = f.getvalue()
            out = out.split('\n')
            # print(f'\n{out[0]}\n{out[1]}')
            if ap_type == 'all':
                ap_95.append((out[0].split('=')[-1]))
                ap_50.append((out[1].split('=')[-1]))
                ap_75.append((out[2].split('=')[-1]))
            if ap_type == 'objarea':
                if areasize == 20:
                    obj_tiny.append((out[3].split('=')[-1]))
                else:
                    obj_small.append((out[3].split('=')[-1]))
                obj_med.append((out[4].split('=')[-1]))
                obj_large.append((out[5].split('=')[-1]))
        
        if ap_type == 'all':
            summary.append(['IoU'] + cat_name + ['mAP'])
            summary.append(['IoU=0.50:0.95'] + ap_95 + [map_95])
            summary.append(['IoU=0.50'] + ap_50 + [map_50])
            summary.append(['IoU=0.75'] + ap_75 + [map_75])

        if ap_type == 'objarea':
            summary.append(['Obj_area'] + cat_name + [f'mAP_{coco_iou}'])
            if areasize == 20:
                summary.append(['tiny'] + obj_tiny + [map_tiny])
            else:    
                summary.append(['small'] + obj_small + [map_small])
            summary.append(['medium'] + obj_med + [map_med])
            summary.append(['large'] + obj_large + [map_large])


        # print('\n|**Summary**|\n')
        # print(summary)
        # print(tabulate(summary, 
        #     headers="firstrow",
        #     tablefmt="psql"))   
        
        if ap_type == 'all': 
            ap_95 = list(map(float, ap_95))
            ap_50 = list(map(float, ap_50))
            # print('IoU=0.50:0.95: ', round(sum(ap_95[:3])/3, 3))
            # print('IoU=0.50: ', round(sum(ap_50[:3])/3,3))

        # if ap_type == 'all':
            # with open(out_csv_filename, 'a') as csv_stat:
            #     csv_stat.write('\n\n')
            #     for s in summary:
            #         for val in s:        
            #             csv_stat.write(f'{val}')

            #             if val != s[-1]:
            #                 csv_stat.write(',')
            #         csv_stat.write('\n')
            # csv_stat.close()
        return summary
  
def build_params(coco_iou, areasize):
    
    params = Params(iouType='bbox')

    if coco_iou == 0.5 or coco_iou == 0.75:
        params.iouThrs = np.array([coco_iou]) 
    # params.iouThrs = np.array([0.50:0.95])
    # params.maxDets = [1, 100, 500]
    params.maxDets = [1, 10, 100]
    # params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, areasize ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    params.areaRngLbl = ['all', 'small', 'medium', 'large']
    params.useCats = 1
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtfile", 
        type=str, 
        help="gt json file path")
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
    args = parser.parse_args()
 
    gt_path = args.gtfile
    pred_path = args.predfile
    output_image = args.outfile
    confusion_matrix_iou_threshold = args.conf_iou

    coco_summary = []    

    ap_type = ['all', 'objarea']

    # for ap in tqdm(ap_type):
    for ap in ap_type:
        
        if ap == 'all':
            coco_iou = 0.95 #set to 0.95 for default coco evaluation
            areasize = 32 #set to 32 for default coco evaluation
            params = build_params(coco_iou, areasize)  # Params for COCO metrics
            performance_evaluation = DetectionPerformanceEvaluation(gt_path, pred_path, params=params,
                                                                    th=confusion_matrix_iou_threshold)
            cls_report = performance_evaluation.build_confussion_matrix(output_image,args.outcsv)
            summary = performance_evaluation.run_coco_metrics(ap, coco_iou, areasize, args.outcsv)
            coco_summary.append(summary)
        
        if ap == 'objarea':
            areasize = [20, 32]

            for a in areasize:

                params = build_params(args.coco_iou, a)  # Params for COCO metrics
                performance_evaluation = DetectionPerformanceEvaluation(gt_path, pred_path, params=params,
                                                                        th=confusion_matrix_iou_threshold)
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
