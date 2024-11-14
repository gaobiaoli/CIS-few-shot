import torch
import clip
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
from ClipAdapter.dataset import CoCoDataset
import argparse
import time
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Save model output to a file with a safe filename.")
    parser.add_argument('--model_name', type=str, required=False, default="ViT-B/32",help="The name of the model (e.g., 'vit_l/14')")
    args = parser.parse_args()
    model_name = args.model_name
    img_path = "/CV/gaobiaoli/dataset/mocs"
    anno_path = "/CV/gaobiaoli/dataset/mocs"
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Zero-shot
    mocstoken = [
        "a photo of a worker",
        "a photo of a tower crane",
        "a photo of a hanging hook",
        "a photo of a vehicle crane",
        "a photo of a roller compactor",
        "a photo of a bulldozer or crawler",
        "a photo of an excavator",
        "a photo of a truck",
        "a photo of a loader",
        "a photo of a concrete pump truck",
        "a photo of a concrete mixer truck",
        "a photo of a pile driver",
        "a photo of a household vehicle",
    ]
    device = "cuda:0"
    clip_model, preprocess = clip.load(model_name, device=device)

    coco_json_test = os.path.join(anno_path, "instances_val.json")
    imgs_path_test = os.path.join(img_path, "images/val")
    dataset_test = CoCoDataset(
        coco_json=coco_json_test,
        imgs_path=imgs_path_test,
        transform=preprocess,
        category_init_id=1,
    )
    dataloader_test = DataLoader(dataset=dataset_test, num_workers=4, batch_size=32)
    all_targets = []
    all_predictions = []
    total_time = 0.0
    total_count = 0

    with torch.no_grad():
        text = clip.tokenize(mocstoken)
        text = text.to(device)
        text_features = clip_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for i, (imgs, label, _) in enumerate(tqdm(dataloader_test)):
            start_time = time.time()

            imgs = imgs.to(device)
            image_features = clip_model.encode_image(imgs)
            image_features_norm = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            logits_per_image = 100.0 * image_features_norm @ text_features.t()

            probs = logits_per_image.softmax(dim=-1).cpu()
            pred_label = probs.argmax(dim=1)

            end_time = time.time()
            iteration_time = end_time - start_time
            total_time += iteration_time

            all_targets.extend(label.cpu().numpy())
            all_predictions.extend(pred_label.cpu().numpy())

    accuracy = metrics.accuracy_score(all_targets, all_predictions)
    precision = metrics.precision_score(all_targets, all_predictions, average=None)
    recall = metrics.recall_score(all_targets, all_predictions, average=None)
    f1 = metrics.f1_score(all_targets, all_predictions, average=None)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. FPS:{:.2f} ****\n".format(accuracy * 100,len(all_targets)/total_time))
    print(precision)
    print(recall)
    print(f1)

    # import pickle
    # with open('zero-shot-b32.pkl', 'wb') as f:
    #     pickle.dump([all_targets, all_predictions], f)
