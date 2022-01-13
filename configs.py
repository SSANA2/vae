# dataset related
dataset_name = "COCO"
resize_h, resize_w = 256, 320
# dataset_name = "MNIST"
# resize_h, resize_w = 28, 28

# dataloader related
batch_size = 48
num_workers = 8
prefetch_factor = 4

# model related
input_ch = 1 if dataset_name == "MNIST" else 3
channels = [input_ch, 32, 64, 128, 256, 512, 1024]
# channels = [input_ch, 32, 64]
num_z = 1024
num_samples = 16

# training related
epochs = 100
init_lr = 5e-4
lr_decay = 0.95

# resize_h, resize_w constraints
modular = 2 ** (len(channels) - 1)
if not ((resize_h % modular == 0) and (resize_w % modular == 0)):
    assert False, "Need to change 'resize_w' or 'resize_h'"