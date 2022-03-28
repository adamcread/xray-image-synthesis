import json
import numpy as np
import math
import matplotlib.pyplot as plt

losses = []
for line in open('out.out', 'r').readlines():
    line = line.rstrip()
    loss = float(json.loads(line)['loss'])
    print(loss)
    if len(losses) > 5:
        # losses.append((sum(losses[-4:]) + loss)/5)
        avg_loss = (sum(losses[-9:])+loss)/10
        if not math.isnan(avg_loss):
            losses.append(avg_loss)
        else:
            losses.append(loss)
    else:
        losses.append(loss)

plt.plot(range(len(losses)), losses)
plt.ylim([0, 1])

plt.savefig('obj_detection_loss.png')