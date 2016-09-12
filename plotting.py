import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import pdb

def plot_training_error(train_err = None, val_err = None, run_name = None, out_file = None):
    plt.figure()
    if train_err is None:
        if val_err is None:
            return
        else:
            n = len(val_err)
    else:
        n = len(train_err)

    leg_list = []
    if train_err is not None:
        plt.plot(range(n), train_err, 'r')
        leg_list.append('train')

    if val_err is not None:
        plt.plot(range(n), val_err, 'b')
        leg_list.append('val')

    plt.legend(leg_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if run_name is not None:
        plt.title('Error Log\n' + run_name)
    else:
        plt.title('Error Log')

    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close()


def compare_images(x, x_hat, save_names, titles=None, cmap='Greys_r', vmin=None, vmax=None):
    '''
    x is like (n_samples, nt, n_channels, nx, ny)
    '''
    is_color = x.shape[2] != 1
    nt = x.shape[1]
    aspect_ratio = float(x_hat.shape[3]) / x_hat.shape[4]
    plt.figure(figsize = (nt, 2*aspect_ratio))
    gs = gridspec.GridSpec(2, nt)
    gs.update(wspace=0., hspace=0.)
    for i in range(x.shape[0]):
        for t in range(nt):
            plt.subplot(gs[t])
            if is_color:
                plt.imshow(np.transpose(x[i,t], (1, 2, 0)), interpolation='none')
            else:
                plt.imshow(x[i,t,0], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
            if t==0:
                if titles is not None: plt.title(titles[i])
                plt.ylabel('Actual', fontsize=10)

            plt.subplot(gs[t + nt])
            if is_color:
                plt.imshow(np.transpose(x_hat[i,t], (1, 2, 0)), interpolation='none')
            else:
                plt.imshow(x_hat[i,t,0], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
            if t==0: plt.ylabel('Predicted', fontsize=10)

        plt.savefig(save_names[i])
        plt.clf()
