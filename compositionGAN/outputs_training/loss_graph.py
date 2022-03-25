import matplotlib.pyplot as plt
import numpy as np
# ['G_GAN', 'D_real', 'D_fake', 'G_L1', 'GP', 'G_seg', 'G_mask', 'STN', 'G_compl', 'D_compl']

def loss_graph(loss_indicators, file):
    loss_values = {loss: [] for loss in loss_indicators}
    name = file.split('.')[0]

    for f in open(file, 'r').readlines():
        f = f.rstrip()
        if f[0] == '(':
            epoch_info, losses_info = f.split(')')
            losses_split = losses_info.strip().split(' ')

            for i in range(0, len(losses_split), 2):
                loss_name = losses_split[i][:-1]
                loss_value = float(losses_split[i+1])

                if loss_name in loss_values.keys():
                    loss_values[loss_name].append(loss_value)
    
    loss_inds = list(loss_indicators.keys())
    fig, axs = plt.subplots(len(loss_indicators))
    fig.set_size_inches(20, 20)
    for i in range(0, len(loss_indicators)):
        axs[i].plot(
            np.array(range(len(loss_values[loss_inds[i]])))/26.719, 
            loss_values[loss_inds[i]]
        )
        axs[i].set_title(f'{loss_inds[i]}')

        bounds = loss_indicators[loss_inds[i]]
        if bounds:
            axs[i].set_xlim([bounds[0], bounds[1]])
            axs[i].set_ylim([bounds[2], bounds[3]])
    
    fig.tight_layout()
    plt.savefig(f'{name}.png')


loss_graph(
    loss_indicators={'G_GAN': [990, 2000, 0, 2],
                     'GP': [990, 2000, 0, 0.1], 
                     'G_L1': [990, 2000, 0, 0.25], 
                     'G_mask': [990, 2000, 0, 0.1], 
                     'G_seg': [], 
                     'D_real': [990, 1100, 0, 1.5], 
                     'D_fake': [990, 1100, 0, 2], 
                     'STN': [], 
                     'G_compl': [490, 600, 0, 25], 
                     'D_compl': [490, 600, 0, 50]
    },
    file='unpaired_combined_dice.txt'
)