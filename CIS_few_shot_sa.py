import os
import clip
import torch
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
from ClipAdapter import CoCoDataset, ClipAdapter
from utils.utils import init_random, get_dataloader
import argparse 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Save model output to a file with a safe filename.")
    parser.add_argument('--model_name', type=str, required=False, default="ViT-B/32",help="The name of the model (e.g., 'vit_l/14')")
    args = parser.parse_args()
    model_name = args.model_name

    init_random(1)
    CIStoken = [
        "a photo of precast component",
        "a photo of a precast component delivery truck",
        "a photo of a bulldozer",
        "a photo of a dump truck",
        "a photo of an excavator",
        "a photo of a concrete mixer",
        "a photo of a person wearing safety helmet correctly",
        "a photo of a person who do not wear safety helmet correctly",
        "a photo of a road roller",
        "a photo of a wheel loader",
    ]
    device = "cuda:0"
    clip_model, preprocess = clip.load(model_name, device=device)

    # shot
    train_tranform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.9, 1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    dataloader_shot,dataloader_val,dataloader_test=get_dataloader('CIS',train_batch_size=4,train_tranform=train_tranform,seed=5200)
    
    clip_adapter = ClipAdapter(
        clip_model,
        device=device,
        dataloader=dataloader_shot,
        classnames=CIStoken,
        alpha=5,
        beta=16,
        augment_epoch=10,
    )

    # val
    clip_adapter.pre_load_features(dataloader=dataloader_val)
    clip_adapter.search_hp(search_scale=[20, 50], search_step=[200, 20],beta_search=True)
    clip_adapter.search_hp(
        search_scale=[20, 51.1], search_step=[200, 102], beta_search=False,print_log=True,inplace=False
    )
    # test
    clip_adapter.pre_load_features(dataloader=dataloader_test)
    
    all_predictions, all_targets, (accuracy, precision, recall, f1) = clip_adapter.eval(
        adapt=False
    )
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(accuracy * 100))
    print(precision)
    print(recall)
    print(f1)
    
    all_predictions, all_targets, (accuracy, precision, recall, f1) = clip_adapter.eval(
        adapt=True
    )
    print("\n**** Few-shot CLIP's val accuracy: {:.2f}. ****\n".format(accuracy * 100))
    print(precision)
    print(recall)
    print(f1)
    # with open("./result/B32-few.pkl","wb") as fp:
    #     pickle.dump([all_predictions,all_targets],fp)