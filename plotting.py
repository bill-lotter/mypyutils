import matplotlib.pyplot as plt
import numpy as np

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


def compare_images(x, x_hat, save_names, titles, cmap='Greys_r', vmin=None, vmax=None):
    '''
    x is like (n_samples, nt, n_channels, nx, ny)
    '''
    if x.shape[2] == 1:
        is_color = False
    else:
        is_color = True
    nt = x.shape[1]
    for i in range(x.shape[0]):
        for t in range(x.shape[1]):
            plt.subplot(2, nt, t+1)
            if is_color:
                plt.imshow(np.transpose(x[i,t], (1, 2, 0)), interpolation='none')
            else:
                plt.imshow(x[i,t,0], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
            plt.axis('off')
            if t==0:
                plt.title(titles[i])

            plt.subplot(2, nt, t+nt+1)
            if is_color:
                plt.imshow(np.transpose(x_hat[i,t], (1, 2, 0)), interpolation='none')
            else:
                plt.imshow(x_hat[i,t,0], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
            plt.axis('off')

            plt.savefig(save_names[i])
