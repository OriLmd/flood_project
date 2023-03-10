
import matplotlib.pyplot as plt

#these functions split the dataset for plotting
def extract_input_parts(input_data, output_data):
    # Split the input tensor into three parts using indexing or slicing
    input_part1 = input_data[:, :, 0:1]
    input_part2 = input_data[:, :, 1:2]
    input_part3 = input_data[:, :, 2:3]
    # Return the input parts and output data as separate tensors
    return (input_part1, input_part2, input_part3), output_data
def extract_input_channels(input,flood):
        return input
def extract_flood_channels(input,flood):
        return flood
def extract_vv(vv,vh,wb):
    return vv
def extract_vh(vv,vh,wb):
    return vh
def extract_wb(vv,vh,wb):
    return wb


#this functions calls the functions above in order to use them to plot
#it returns 4 channels that can be used to plot stuff
def split_tensor_channel(dataset):
    all_channels = dataset.map(extract_input_parts)
    flood_channel = all_channels.map(extract_flood_channels)
    input_channels = all_channels.map(extract_input_channels)
    vv_channel = input_channels.map(extract_vv)
    vh_channel = input_channels.map(extract_vh)
    wb_channel = input_channels.map(extract_wb)
    return vv_channel,vh_channel,wb_channel,flood_channel

# Plot function
def plot_results(vv_channel, vh_channel, wb_channel, flood_channel, predictions, threshold = 0.5, n_img = 8, start_num = 0):
    '''
    This function is ploting the inputs, the targets and the predictions found by the model
    '''

    # Apply a threshold to the prediction and apply 0 or 1 if it's water or land
    predictions_binary = predictions.copy()
    predictions_binary[predictions_binary < threshold] = 0
    predictions_binary[predictions_binary > threshold] = 1

    # Define the fig size, the main title and the rows titles names
    fig, axes = plt.subplots(nrows=5, ncols=n_img, figsize=(20, 14))
    fig.suptitle('Inputs and Predictions', fontsize=16)

    row_titles = ['VV polarization', 'VH polarization', 'Water body', 'Targets : Flood', 'Predictions : Flood']

    # Ploting each image (sample of n_img different images)
    for i, (vv, vh, wb, flood, pred) in enumerate(zip(vv_channel.skip(start_num).take(n_img),
                                                    vh_channel.skip(start_num).take(n_img),
                                                    wb_channel.skip(start_num).take(n_img),
                                                    flood_channel.skip(start_num).take(n_img),
                                                    predictions_binary[start_num:])):

        axes[0, i].imshow(vv.numpy(), cmap='Greys_r')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[0, i].set_title(f'vv {i+1+start_num}', fontsize=12)

        axes[1, i].imshow(vh.numpy(), cmap='Greys_r')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        axes[1, i].set_title(f'vh {i+1+start_num}', fontsize=12)

        axes[2, i].imshow(wb.numpy(), cmap='Greys')
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])
        axes[2, i].set_title(f'wb {i+1+start_num}', fontsize=12)

        axes[3, i].imshow(flood.numpy(), cmap='Greys')
        axes[3, i].set_xticks([])
        axes[3, i].set_yticks([])
        axes[3, i].set_title(f'Flood {i+1+start_num}', fontsize=12)

        axes[4, i].imshow(pred, cmap='Greys')
        axes[4, i].set_xticks([])
        axes[4, i].set_yticks([])
        axes[4, i].set_title(f'Prediction {i+1+start_num}', fontsize=12)

    # Plot the rows titles
    for i in range(5):
        axes[i, 0].set_ylabel(row_titles[i], fontsize=14, rotation=90, labelpad=20)

    # Plot the whole fig
    plt.tight_layout()
    plt.show()

#transformer les r??sultats de proba en binaire O 1 1 ??tant l'eau
def result_into_class(y,threshold=0.5):
    class_view = y
    class_view[class_view >= threshold] = 1
    class_view[class_view < threshold] = 0
    return class_view


# lui mettre deux images (array) renvoie les false positive en bleu, false negative en rouge, true positif en gris fonc?? et true negatif en gris clair
def return_fn_fp_image(img_flood,img_river):

    C = np.zeros(shape=(len(img_flood), len(img_flood[0]), 3))

    for i in range (0, img_river.shape[0],1):
        for j in range(0, img_river.shape[1], 1):
            # if img_river[i][j] == img_flood[i][j] and img_river[i][j] == 0:
            #     C[i][j] = 1
            if img_river[i][j] == 0 and img_flood[i][j]==0:
                C[i][j][0] = 0.8
                C[i][j][1] = 0.8
                C[i][j][2] = 0.8
            elif img_river[i][j] == 1 and img_flood[i][j]==1:
                C[i][j][0] = 0.5
                C[i][j][1] = 0.5
                C[i][j][2] = 0.5
            elif img_river[i][j] == 0 and img_flood[i][j]==1:
                C[i][j][0] = 0.26
                C[i][j][1] = 0.55
                C[i][j][2] = 0.96

            else:
                C[i][j][0] = 1
                C[i][j][1] = 0
                C[i][j][2] = 0

    plt.imshow(C)


# lui mettre deux images (array) et renvoie l'interpr??tation des zones inond??es en bleu clair, zones de rivi??re en bleu et zone de terre en vert
def return_flood_image(img_flood,img_river):

    C = np.zeros(shape=(len(img_flood), len(img_flood[0]), 3))

    for i in range (0, img_river.shape[0],1):
        for j in range(0, img_river.shape[1], 1):
            # if img_river[i][j] == img_flood[i][j] and img_river[i][j] == 0:
            #     C[i][j] = 1
            if img_river[i][j] == 0 and img_flood[i][j]==0:
                C[i][j][0] = 0.05
                C[i][j][1] = 0.5
                C[i][j][2] = 0.11
            elif img_river[i][j] == 1 and img_flood[i][j]==1:
                C[i][j][0] = 0
                C[i][j][1] = 0
                C[i][j][2] = 1
            elif img_river[i][j] == 0 and img_flood[i][j]==1:
                C[i][j][0] = 0.26
                C[i][j][1] = 0.55
                C[i][j][2] = 0.96

            else:
                C[i][j][0] = 0
                C[i][j][1] = 0
                C[i][j][2] = 1

    plt.imshow(C)
