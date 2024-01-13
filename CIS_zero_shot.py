import torch
import clip
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
from ClipAdapter.dataset import CoCoDataset

if __name__ == "__main__":
    img_path = "/CV/gaobiaoli/dataset/CIS-Dataset"
    anno_path = "/CV/gaobiaoli/dataset/CIS-Dataset/dataset/annotations"
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Zero-shot
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
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    coco_json_test = os.path.join(anno_path, "test.json")
    imgs_path_test = os.path.join(img_path, "test")
    dataset_test = CoCoDataset(
        coco_json=coco_json_test,
        imgs_path=imgs_path_test,
        transform=preprocess,
        category_init_id=0,
    )
    dataloader_test = DataLoader(dataset=dataset_test, num_workers=4, batch_size=32)
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        text = clip.tokenize(CIStoken)
        text = text.to(device)
        text_features = clip_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for i, (imgs, label, _) in enumerate(tqdm(dataloader_test)):
            imgs = imgs.to(device)
            image_features = clip_model.encode_image(imgs)
            image_features_norm = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            logits_per_image = 100.0 * image_features_norm @ text_features.t()

            probs = logits_per_image.softmax(dim=-1).cpu()
            pred_label = probs.argmax(dim=1)

            all_targets.extend(label.cpu().numpy())
            all_predictions.extend(pred_label.cpu().numpy())

    accuracy = metrics.accuracy_score(all_targets, all_predictions)
    precision = metrics.precision_score(all_targets, all_predictions, average=None)
    recall = metrics.recall_score(all_targets, all_predictions, average=None)
    f1 = metrics.f1_score(all_targets, all_predictions, average=None)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(accuracy * 100))
    print(precision)
    print(recall)
    print(f1)
