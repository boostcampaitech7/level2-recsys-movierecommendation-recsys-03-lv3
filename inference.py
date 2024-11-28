import os
import argparse

from omegaconf import OmegaConf

import src.loader as loader_module
import src.model as model_module
import src.trainer as trainer_module
from src.utils import set_seed, check_path, save_recommendations


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", "--m", default="DeepFM", type=str,
                        help="사용할 모델을 설정할 수 있습니다. (기본값 DeepFM)")
    parser.add_argument("--run", "--r", type=str,
                        help="불러올 .pt 파일의 모델명 뒤 run명을 설정할 수 있습니다.")
    parser.add_argument("--device", "--d", default="cuda", type=str,
                        choices=["cuda", "cpu"], help="device를 설정할 수 있습니다. (기본값 cuda)")

    args = parser.parse_args()

    config = "config/config_baseline.yaml"
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(config) if config else OmegaConf.create()

    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]

    args = config_yaml
    args_str = f"{args.model_name}_{args.run}"
    checkpoint = args_str + ".pt"
    checkpoint_path = os.path.join(args.output_path, checkpoint)

    set_seed(args.seed)
    check_path(args.output_path)

    print("----------------------- LOAD DATA -----------------------")
    _, _, submission_loader, seen_items, idx_to_user, idx_to_item, _ = getattr(loader_module, args.model_name)(args).load_data()

    print(f"--------------------- INIT {args.model_name} ----------------------")
    model = getattr(model_module, args.model_name)(**args.model_args[args.model_name]).to(args.device)

    print(f"-------------------- {args.model_name} PREDICT --------------------")
    trainer = getattr(trainer_module, args.model_name)(model, None, None, submission_loader, seen_items, args)

    trainer.load(checkpoint_path)
    recommendations = trainer.submission(0)

    save_recommendations(recommendations, idx_to_user, idx_to_item, args.model_name, args.output_path)


if __name__ == "__main__":
    main()
