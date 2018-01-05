import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import pdb
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

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


def plot_ranked_ims(y, y_hat, x, n_plot, save_dir):
    idx = np.argsort(y_hat.flatten())
    y_hat = y_hat[idx]
    y = y[idx]
    x = x[idx]
    n_plot = min(n_plot, len(y)/2)

    plt.figure()
    for i in range(-n_plot, n_plot):
        plt.imshow(x[i,0], cmap="Greys_r", interpolation='none')
        plt.title('rank='+str(i)+' y='+str(y[i])+' yhat='+str(y_hat[i]))
        plt.axis('off')
        plt.savefig(save_dir+'plot_'+str(i)+'.png')

    plt.close('all')


def plot_roc(y, y_hat, save_name, title_str):

    fpr, tpr, roc_thresholds = roc_curve(y, y_hat)
    roc_auc = roc_auc_score(y, y_hat)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title_str)
    plt.legend(loc="lower right")
    plt.savefig(save_name)

    plt.close('all')
    return roc_auc


def plot_precision_recall(y, y_hat, save_name, title_str):
    precision, recall, pr_thresholds = precision_recall_curve(y, y_hat)
    ap_score = average_precision_score(y, y_hat)

    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall AUC={0:0.2f}'.format(ap_score))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title(title_str)
    plt.legend(loc="lower left")
    plt.savefig(save_name)

    plt.close()
    return ap_score
