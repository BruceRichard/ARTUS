from utils import parse_config_from_args
from lightning.pytorch import seed_everything
from model.Transformer.eval import Evaluater
from utils.mylogging import Log
from pathlib import Path

from rich import print
from tqdm import tqdm

import multiprocessing
import time
import random
import hashlib

if __name__ == '__main__':
    config = parse_config_from_args()
    Log.info(f'Loading : {Evaluater}')
    evaluator = Evaluater(config)

    text_datasets = Path('data/datasets/3_text_condition')

    obj_paths = list(text_datasets.glob('*'))
    random.shuffle(obj_paths)
    tt = time.strftime("%m-%d-%I%p-%M-%S")

    multiprocessing.set_start_method("spawn")

    OPTION = config['OPTION']
    if OPTION == 1:
        # obj_name_list_all = list(map(lambda x : x.parent.stem + '_' + x.stem,
        #                              Path('data/datasets/3_text_condition').glob('*/*')))
        # obj_name_list = random.choices(obj_name_list_all, k=100)
        # random.shuffle(obj_name_list)

        # target_folder_name = "StorageFurniture_45940"
        # target_dir = text_datasets / target_folder_name
        # if not target_dir.exists():
        #     Log.error(f"Target directory does not exist: {target_dir}")
        #     exit(1)

        # target_files = list(target_dir.glob('*.txt'))
        
        # obj_name_list = []
        # for f in target_files:
        #     # 拼接格式：StorageFurniture_45940 + '_' + 0
        #     formatted_name = f"{f.parent.stem}_{f.stem}"
        #     obj_name_list.append(formatted_name)

        # print(f"Found {len(obj_name_list)} text files in {target_folder_name}")

        generation_cfg = config.get('generation', {})
        target_category_keyword = generation_cfg.get('category_keyword', "Table")
        num_objects = int(generation_cfg.get('num_objects', 100))
        fixed_experiment_seed = int(generation_cfg.get('selection_seed', 31))
        repetitions = int(generation_cfg.get('repetitions', 3))
        experiment_name = generation_cfg.get('experiment_name', 'artus_full')
        output_root = Path(generation_cfg.get('output_root', 'elog/final_output'))

        candidate_dirs = []
        for d in text_datasets.iterdir():
            if d.is_dir() and target_category_keyword in d.name:
                candidate_dirs.append(d)
        candidate_dirs.sort(key=lambda x: x.name)

        rng = random.Random(fixed_experiment_seed)
        if len(candidate_dirs) > num_objects:
            selected_dirs = rng.sample(candidate_dirs, num_objects)
        else:
            Log.warning(f"Found only {len(candidate_dirs)} directories, less than target {num_objects}. Using all.")
            selected_dirs = candidate_dirs
        obj_name_list = []

        for d in selected_dirs:
            txt_files = sorted(d.glob('*.txt'), key=lambda f: int(f.stem))  # 0.txt, 1.txt...
            for f in txt_files:
                formatted_name = f"{f.parent.stem}_{f.stem}"
                obj_name_list.append(formatted_name)

        for obj_name in tqdm(obj_name_list, 'obj_list'):
            output_path = output_root / experiment_name / obj_name
            obj_infos = obj_name.split('_')
            text_content = (text_datasets / '_'.join(obj_infos[:2]) / (str(obj_infos[2])+'.txt')).read_text()
            print("Processing", obj_name)

            for rep in range(repetitions):
                seed_material = f"{fixed_experiment_seed}:{obj_name}:{rep}".encode("utf-8")
                sample_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "little")
                seed_everything(sample_seed, workers=True)
                evaluator.inference_to_output_path(text_content, output_path / str(rep), blender_generated_gif=True)
    else:
        print('NOT SUPPORT ANYMORE.')
