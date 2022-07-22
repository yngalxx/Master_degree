from controller import controller
import pathlib
import click

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--channel', 
    default=1, type=int, 
    help='Image channels: 3 <= RGB, 1 <= greyscale.', 
    show_default=True
)
@click.option(
    '--num_classes', 
    default=8, type=int, 
    help='Number of classes.', 
    show_default=True
)
@click.option(
    '--learning_rate', 
    default=3e-4, type=float, 
    help='Learning rate value.', 
    show_default=True
)
@click.option(
    '--batch_size', 
    default=16, type=int, 
    help='Number of batches.', 
    show_default=True
)
@click.option(
    '--num_epochs', 
    default=20, type=int, 
    help='Number of epochs.', 
    show_default=True
)
@click.option(
    '--rescale', 
    default='1000/1000', type=str, 
    help='2 possible ways to rescale your images and also annotations. First one is by using following \
        pattern "width/height" and then each image will be scaled to that size, thanks to it you will \
        have every image in the same size (less computational complexity). The other way is to enter a \
        float value (however you still have to put it in a string i.e. ".5" and value has to be bigger \
        than 0 and smaller or equal than 1), then each image will be multiplied by this value. If you \
        pass 1 as a value images and annotations will not be scaled.',
    show_default=True
)
@click.option(
    '--shuffle', 
    default=False, type=bool, 
    help='Shuffle data.', 
    show_default=True
)
@click.option(
    '--weight_decay', 
    default=0, type=float, 
    help='Weight decay regularization value.', 
    show_default=True
)
@click.option(
    '--lr_scheduler', 
    default=True, type=bool, 
    help='Learning rate scheduler: if value=True learning rate scheduler will be enabled, if value=False \
        it will be disabled.', 
    show_default=True
)
@click.option(
    '--lr_step_size', 
    default=5, type=int, 
    help='Step size of learning rate scheduler: valid only if learning rate scheduler is enabled.', 
    show_default=True
)
@click.option(
    '--lr_gamma', 
    default=.4, type=float, 
    help='Valid only when learning rate scheduling is enabled, passed value determines the learning \
    rate multiplier.', 
    show_default=True
)
@click.option(
    '--trainable_backbone_layers', 
    default=5, type=int, 
    help='Number of trainable layers in pretrained ResNet-50 network.', 
    show_default=True
)
@click.option(
    '--num_workers', 
    default=2, type=int, 
    help='Setting the argument num_workers as a positive integer will turn on multi-process data \
        loading with the specified number of loader worker processes.', 
    show_default=True
)
@click.option(
    '--main_dir', 
    default=f'{"/".join(str(pathlib.Path(__file__).parent.resolve()).split("/")[:-1])}/', type=str, 
    help='Working directory path.', 
    show_default=True
)
@click.option(
    '--image_dir', 
    default=f'{"/".join(str(pathlib.Path(__file__).parent.resolve()).split("/")[:-2])}/scraped_photos/', type=str, 
    help='Image directory path.', 
    show_default=True
)
@click.option(
    '--annotations_dir', 
    default=f'{"/".join(str(pathlib.Path(__file__).parent.resolve()).split("/")[:-2])}/news-navigator/', type=str, 
    help='Preprocessed annotations directory path.', 
    show_default=True
)
@click.option(
    '--train', 
    default=True, type=bool, 
    help='Train the model', 
    show_default=True
)
@click.option(
    '--predict', 
    default=True, type=bool, 
    help='Make prediction.', 
    show_default=True
)
@click.option(
    '--train_set', 
    default=True, type=bool, 
    help='Use training data set.',
    show_default=True
)
@click.option(
    '--test_set', 
    default=True, type=bool, 
    help='Use test data set.', 
    show_default=True
)
@click.option(
    '--val_set', 
    default=True, type=bool, 
    help='Use validation data set.', 
    show_default=True
)
@click.option(
    '--gpu', 
    default=True, type=bool, 
    help='Enable training on GPU.', 
    show_default=True
)
@click.option(
    '--bbox_format', 
    default='x0y0x1y1', type=str, 
    help='Bounding boxes format. Other allowed format is "x0y0wh", where w - width and h - height.', 
    show_default=True
)


def main( 
    channel, num_classes, learning_rate, batch_size, num_epochs, rescale, shuffle, weight_decay,
    lr_scheduler, lr_step_size, lr_gamma, trainable_backbone_layers, num_workers, main_dir, 
    image_dir, annotations_dir, train, predict, train_set, test_set, val_set, gpu, bbox_format
):  
    controller(
        channel=channel, 
        num_classes=num_classes, 
        learning_rate=learning_rate, 
        batch_size=batch_size, 
        num_epochs=num_epochs, 
        rescale=rescale, 
        shuffle=shuffle, 
        weight_decay=weight_decay,
        lr_scheduler=lr_scheduler, 
        lr_step_size=lr_step_size, 
        lr_gamma=lr_gamma,
        trainable_backbone_layers=trainable_backbone_layers, 
        num_workers=num_workers, 
        main_dir=main_dir, 
        image_dir=image_dir, 
        annotations_dir=annotations_dir, 
        train=train, 
        predict=predict, 
        train_set=train_set, 
        test_set=test_set, 
        val_set=val_set, 
        gpu=gpu, 
        bbox_format=bbox_format
    )


if __name__ == '__main__':
    main()
