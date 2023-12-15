import os
import torch
import albumentations

import numpy as np
import pandas as pd

import torch.nn as nn
from sklearn import metrics
from sklearn import model_selection
from torch.nn import functional as F

from wtfml.utils import EarlyStopping
from wtfml.engine import Engine
from wtfml.data_loaders.image import ClassificationLoader

import pretrainedmodels


class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained='imagenet', model_dir_path=''):
        super(SEResnext50_32x4d, self).__init__()
        
        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=None)
        if pretrained is not None:
            self.base_model.load_state_dict(
                torch.load(
                    model_dir_path+"/se_resnext50_32x4d-a260b3a4.pth"
                )
            )

        self.out = nn.Linear(2048, 1)
    
    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape
        
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        
        out = self.out(x)
 #       out = torch.sigmoid(out)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(out))

        return out, loss


class SkinCancerModel:

    def __init__(self, model_dir_path, model_name_prefix) -> None:
        if os.listdir(model_dir_path) == 0:
            print("Model Dir is empty that you have provided")

        self.model_dir_path = model_dir_path
        self.model_name_prefix = model_name_prefix
        self.device_name = "cpu"

    def prediction_per_fold(self, fold, local_image_path):
    
        fold_model_path = self.model_dir_path + "/" +self.model_name_prefix + fold + ".bin"

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        aug = albumentations.Compose(
            [
                albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
            ]
        )

        images = [local_image_path]
        targets = np.zeros(len(images))

        test_dataset = ClassificationLoader(
            image_paths=images,
            targets=targets,
            resize=None,
            augmentations=aug,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=16, shuffle=False, num_workers=4
        )

        print(fold_model_path)
        model = SEResnext50_32x4d(pretrained=None, model_dir_path=self.model_dir_path )
        # model.load_state_dict(torch.load(fold_model_path))
        # model.to(self.device)
        device = torch.device(self.device_name)
        model.to(device)    
        model.load_state_dict(torch.load(fold_model_path, map_location=device))
        

        predictions = Engine.predict(test_loader, model
                                    , device=self.device_name
                                        )
        predictions = np.vstack((predictions)).ravel()
        predictions = torch.sigmoid(torch.tensor(predictions)).cpu().numpy()
        first_value = predictions[0] if predictions.size > 0 else None
    
        return first_value
        

    def predict_for_image(self, local_image_path):
        prediction_0 = self.prediction_per_fold("0", local_image_path)
        prediction_1 = self.prediction_per_fold("1", local_image_path)
        prediction_2 = self.prediction_per_fold("2", local_image_path)
        prediction_3 = self.prediction_per_fold("3", local_image_path)
        prediction_4 = self.prediction_per_fold("4", local_image_path)

        return (prediction_0 + prediction_1 + prediction_2 + prediction_3 + prediction_4) / 5