import torch
from model.engine import train_one_epoch, evaluate
from model.dataset import SyntheticSPMDataset
from model import utils
from model import detection
import albumentations as A

import os
from dataclasses import dataclass


@dataclass
class ModelConfiguration:
    shape: tuple[int, int]
    J: int
    L: int
    num_classes: int


@dataclass
class Configuration:
    start_epoch: int
    epochs: int
    resume: bool
    output_dir: str
    batch_size: int


# ---------- RUN CONFIG ----------- #
args = Configuration(start_epoch=0,
                     epochs=2,
                     resume=False,
                     output_dir='./model/result/',
                     batch_size=2)  # was 2


# ---------- MODEL CONFIG ---------- #
model_config = ModelConfiguration(
    shape=(256, 512),
    J=3, L=8, num_classes=2)
# our dataset has two classes only - background and person
#

# ---------- DATASET CONFIG ---------- #
DATASET_CONFIG = {
    'surfaces_dir': './syn/',
    'objects_dir': './objects/',
    'max_objects': 10,
    'object_scale_limit': [0.9, 1.1],
    'object_rot_limit': [-90, 90],
    'img_ext': '.png',
    'transform': None
    # A.Compose(
    #     [
    #         A.RandomCrop(width=450, height=450),
    #         A.HorizontalFlip(p=0.5),
    #         A.RandomBrightnessContrast(p=0.2),
    #     ], bbox_params=A.BboxParams(format='coco'))
}


def get_model_instance_segmentation() -> torch.nn.Module:
    return detection.Scattering2DSSD(shape=model_config.shape,
                                     J=model_config.J,
                                     L=model_config.L,
                                     num_classes=model_config.num_classes)


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# use our dataset and defined transformations
dataset = SyntheticSPMDataset(**DATASET_CONFIG)
dataset_test = SyntheticSPMDataset(**DATASET_CONFIG)

# split the dataset in train and test set
#train_sampler = torch.utils.data.RandomSampler(dataset)
#test_sampler = torch.utils.data.SequentialSampler(dataset_test)
#train_batch_sampler = torch.utils.data.BatchSampler(
#    train_sampler, args.batch_size, drop_last=True)

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    #batch_sampler=train_batch_sampler,
    batch_size=2,
    shuffle=True,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    #batch_sampler=test_sampler,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

# get the model using our helper function
model = get_model_instance_segmentation()

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu')
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    args.start_epoch = checkpoint['epoch'] + 1

for epoch in range(args.start_epoch, args.epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

    if args.output_dir:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'args': args,
            'epoch': epoch
        }
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'checkpoint.pth'))

print("That's it!")
