
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
