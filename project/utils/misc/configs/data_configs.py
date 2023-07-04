from . import transforms_config
from .paths_config import dataset_paths

DATASETS = {
    'ffhq_encode_renderer': {
        'transforms': transforms_config.EncodeTransforms_lowRes,
        'train_source_root': dataset_paths['ffhq'],
        'train_target_root': dataset_paths['ffhq'],
        'test_source_root': dataset_paths['celeba_test'],
        'test_target_root': dataset_paths['celeba_test']
    },
    'ffhq_encode': {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['ffhq'],
        'train_target_root': dataset_paths['ffhq'],
        'test_source_root': dataset_paths['celeba_test'],
        'test_target_root': dataset_paths['celeba_test']
    },
    "cars_encode": {
        'transforms': transforms_config.CarsEncodeTransforms,
        'train_source_root': dataset_paths['cars_train'],
        'train_target_root': dataset_paths['cars_train'],
        'test_source_root': dataset_paths['cars_test'],
        'test_target_root': dataset_paths['cars_test']
    },
    "church_encode": {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['church_train'],
        'train_target_root': dataset_paths['church_train'],
        'test_source_root': dataset_paths['church_test'],
        'test_target_root': dataset_paths['church_test']
    },
    "horse_encode": {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['horse_train'],
        'train_target_root': dataset_paths['horse_train'],
        'test_source_root': dataset_paths['horse_test'],
        'test_target_root': dataset_paths['horse_test']
    },
    "afhq_wild_encode": {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['afhq_wild_train'],
        'train_target_root': dataset_paths['afhq_wild_train'],
        'test_source_root': dataset_paths['afhq_wild_test'],
        'test_target_root': dataset_paths['afhq_wild_test']
    },
    "toonify": {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['ffhq'],
        'train_target_root': dataset_paths['ffhq'],
        'test_source_root': dataset_paths['celeba_test'],
        'test_target_root': dataset_paths['celeba_test']
    }
}
