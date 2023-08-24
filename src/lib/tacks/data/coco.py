# -*- coding: utf-8 -*-
"""COCO dataset utils.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import json
import random
import shutil

import torchvision
from tqdm import tqdm

from tacks.utils import get_config, set_seed


def create_subdataset(split_name, coco_size, seed=None):
    """Create a sub-dataset of the COCO dataset.

    Samples are randomly drawn from the selected split. A dataset 'coco{coco_size}' is
    created in the data folder.

    Parameters
    ----------
    split_name : str
        Name of the split.
    coco_size : int
        Size of the COCO subdataset.
    """

    if seed is not None:
        set_seed(seed)

    config = get_config()

    data_path = config.get_path('paths', 'data')
    coco_path = data_path / 'coco'
    subcoco_path = data_path / f'coco{coco_size}'

    subcoco_path.mkdir(exist_ok=True)
    (subcoco_path / 'annotations').mkdir(exist_ok=True)

    if (subcoco_path / 'annotations' / f'{split_name}.json').exists():
        raise ValueError(
            f'Annotations for `coco{coco_size}.{split_name}` already exist.'
        )

    print('Loading annotations...', end='')
    with open(
        coco_path / 'annotations' / f'instances_{split_name}.json', 'r'
    ) as infile:
        coco_anns = json.load(infile)
    print('Done.')

    subcoco_anns = dict()

    subcoco_anns['info'] = coco_anns['info'].copy()
    subcoco_anns['info']['description'] += ' - Variant {n_samples}'
    subcoco_anns['licenses'] = coco_anns['licenses'].copy()
    subcoco_anns['categories'] = coco_anns['categories'].copy()
    subcoco_anns['images'] = [
        img_info for img_info in random.sample(coco_anns['images'], coco_size)
    ]

    print('Picking images...', end='')
    img_ids = [img_info['id'] for img_info in subcoco_anns['images']]
    print('Done.')

    print('Getting annotations...', end='')
    subcoco_anns['annotations'] = [
        annotation
        for annotation in coco_anns['annotations']
        if annotation['image_id'] in img_ids
    ]
    print('Done.')

    print('Saving annotations...', end='')
    minicoco_path = data_path / f'coco{coco_size}'
    minicoco_path.mkdir(exist_ok=True)
    (minicoco_path / 'annotations').mkdir(exist_ok=True)

    if (minicoco_path / 'annotations' / f'instances_{split_name}.json').exists():
        raise ValueError(
            f'Annotations for `coco{coco_size}.{split_name}` already exist.'
        )

    with open(
        minicoco_path / 'annotations' / f'instances_{split_name}.json', 'w'
    ) as outfile:
        json.dump(subcoco_anns, outfile)
    print('Done.')

    # copy images
    print('Copying images...', end='')
    (minicoco_path / split_name).mkdir(exist_ok=True)
    for img_info in subcoco_anns['images']:

        shutil.copy(
            coco_path / split_name / img_info['file_name'],
            minicoco_path / split_name / img_info['file_name'],
        )
    print('Done.')


def convert_coco_dataset(split_name, coco_size):
    """Convert a coco split into a suitable format for `DetectionDataset`.

    Parameters
    ----------
    split_name : str
        Name of the split.
    coco_size : int
        Size of the COCO dataset.
    """

    data_path = get_config().get_path('paths', 'data') / f'coco{coco_size}'

    dataset = torchvision.datasets.CocoDetection(
        root=data_path / split_name,
        annFile=data_path / 'annotations' / f'instances_{split_name}.json',
    )

    dataset_path = dataset.root.parent
    split_name = dataset.root.stem

    # create directories inside dataset folder
    img_paths = dataset_path / 'images' / split_name
    label_paths = dataset_path / 'labels' / split_name

    img_paths.mkdir(exist_ok=True, parents=True)
    label_paths.mkdir(exist_ok=True, parents=True)

    # copy images in folder
    pb_images = tqdm(dataset.coco.imgs.items())

    for img_id, img_info in pb_images:

        pb_images.desc = f'Processing image {img_id}...'
        img_path = dataset.root / img_info['file_name']
        shutil.copy(img_path, img_paths / img_path.name)

        annotations = dataset._load_target(img_id)

        with open(label_paths / f'{img_path.stem}.txt', 'w') as outfile:

            for annotation in annotations:
                ann = annotation['bbox'] + [annotation['category_id']]
                outfile.write(' '.join([str(item) for item in ann]))
                outfile.write('\n')
