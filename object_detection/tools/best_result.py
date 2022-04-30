import argparse
import os

parser = argparse.ArgumentParser(description='Best result from log file')
parser.add_argument('--log_root', help='train log file path')

args = parser.parse_args()

log_file = [x for x in os.listdir(args.log_root) if x.endswith(".log")][0]

results = []
logs = open(args.log_root+log_file, 'r').readlines()
for i, log in enumerate(logs):
    log = log.rstrip().split('\\n')[0]

    if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]' in log:
        result_05_09 = float(log.split('=')[-1])
        result_05 = float(logs[i+1].split('\\n')[0].split('=')[-1])
        results.append([result_05_09 + result_05, result_05_09, result_05])

    
best_result = max(results, key=lambda x: x[2])
print('0.5-0.9: ', best_result[1])
print('0.5: ', best_result[2])
print('sum', best_result[0])
