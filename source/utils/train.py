import matplotlib.pyplot as plt
import shutil
import torch


def save_checkpoint(state, is_best, checkpoint_path, filename='checkpoint.pth.tar'):
    best_val = []
    best_val.append(state['best_acc'])
    torch.save(state, checkpoint_path + filename)
    if is_best:
        shutil.copyfile(checkpoint_path + filename,
                        checkpoint_path + 'model_best.pth.tar')
        print('\tAccuracy is updated and the params is saved in [model_best.pth.tar]!'.ljust(20), flush=True)


def show(atlas, img, pred, img_label, atlas_label_slice, pred_label_slice):
    fig, ax = plt.subplots(2, 3)
    fig.dpi = 200

    ax0 = ax[0][0].imshow(atlas, cmap='gray')
    ax[0][0].set_title('atlas')
    ax[0][0].axis('off')

    ax1 = ax[0][1].imshow(img, cmap='gray')
    ax[0][1].set_title('moving')
    ax[0][1].axis('off')

    ax2 = ax[0][2].imshow(pred, cmap='gray')
    ax[0][2].set_title('pred')
    ax[0][2].axis('off')

    ax4 = ax[1][0].imshow(atlas_label_slice, cmap='tab20')
    ax[1][0].set_title('atlas_label')
    ax[1][0].axis('off')

    ax4 = ax[1][1].imshow(img_label, cmap='tab20')
    ax[1][1].set_title('moving_label')
    ax[1][1].axis('off')

    ax5 = ax[1][2].imshow(pred_label_slice, cmap='tab20')
    ax[1][2].set_title('pred_label')
    ax[1][2].axis('off')

    return fig
