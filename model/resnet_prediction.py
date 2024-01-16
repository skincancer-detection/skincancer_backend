from genericpath import isdir
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchtoolbox.transform as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import pandas as pd
import numpy as np
import os
import cv2
import warnings
import random
from efficientnet_pytorch import EfficientNet


current_dir = os.getcwd()

warnings.simplefilter('ignore')
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(47)
device = 'cpu'


class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, file_path: str, train: bool = True, transforms = None, meta_features = None):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            file_path (str): folder with images
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            meta_features (list): list of features with meta information, such as sex and age
            
        """
        self.df = df
        self.file_path = file_path
        self.transforms = transforms
        self.train = train
        self.meta_features = meta_features
        
    def __getitem__(self, index):
        x = cv2.imread(self.file_path)
        meta = np.array(self.df.iloc[index][self.meta_features].values, dtype=np.float32)

        if self.transforms:
            x = self.transforms(x)
            
        if self.train:
            y = self.df.iloc[index]['target']
            return (x, meta), y
        else:
            return (x, meta)
    
    def __len__(self):
        return len(self.df)
    
    
class Net(nn.Module):
    def __init__(self, arch, n_meta_features: int):
        super(Net, self).__init__()
        self.arch = arch
        if 'ResNet' in str(arch.__class__):
            self.arch.fc = nn.Linear(in_features=512, out_features=500, bias=True)
        if 'EfficientNet' in str(arch.__class__):
            self.arch._fc = nn.Linear(in_features=1280, out_features=500, bias=True)
        self.meta = nn.Sequential(nn.Linear(n_meta_features, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(500, 250),  # FC layer output will have 250 features
                                  nn.BatchNorm1d(250),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        self.ouput = nn.Linear(500 + 250, 1)
        
    def forward(self, inputs):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x, meta = inputs
        cnn_features = self.arch(x)
        meta_features = self.meta(meta)
        features = torch.cat((cnn_features, meta_features), dim=1)
        output = self.ouput(features)
        return output


class AdvancedHairAugmentation:
    """
    Impose an image of a hair to the target image

    Args:
        hairs (int): maximum number of hairs to impose
        hairs_folder (str): path to the folder with hairs images
    """

    def __init__(self, hairs: int = 5, hairs_folder: str = ""):
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        n_hairs = random.randint(0, self.hairs)
        
        if not n_hairs:
            return img
        
        height, width, _ = img.shape  # target image width and height
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]
        
        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            # Creating a mask and inverse mask
            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of hair in ROI
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of hair from hair image.
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            # Put hair in ROI and modify the target image
            dst = cv2.add(img_bg, hair_fg)

            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst
                
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, hairs_folder="{self.hairs_folder}")'


class DrawHair:
    """
    Draw a random number of pseudo hairs

    Args:
        hairs (int): maximum number of hairs to draw
        width (tuple): possible width of the hair in pixels
    """

    def __init__(self, hairs:int = 4, width:tuple = (1, 2)):
        self.hairs = hairs
        self.width = width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        if not self.hairs:
            return img
        
        width, height, _ = img.shape
        
        for _ in range(random.randint(0, self.hairs)):
            # The origin point of the line will always be at the top half of the image
            origin = (random.randint(0, width), random.randint(0, height // 2))
            # The end of the line 
            end = (random.randint(0, width), random.randint(0, height))
            color = (0, 0, 0)  # color of the hair. Black.
            cv2.line(img, origin, end, color, random.randint(self.width[0], self.width[1]))
        
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, width={self.width})'


class Microscope:
    """
    Cutting out the edges around the center circle of the image
    Imitating a picture, taken through the microscope

    Args:
        p (float): probability of applying an augmentation
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to apply transformation to.

        Returns:
            PIL Image: Image with transformation.
        """
        if random.random() < self.p:
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8), # image placeholder
                        (img.shape[0]//2, img.shape[1]//2), # center point of circle
                        random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15), # radius
                        (0, 0, 0), # color
                        -1)

            mask = circle - 255
            img = np.multiply(img, mask)
        
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


class PredictHelper:
    def prediction_helper(self, file_path, user_metadata={}):

        train_transform = transforms.Compose([
            AdvancedHairAugmentation(hairs_folder=f'{current_dir}/melanoma-hairs'),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            Microscope(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        
        data_dict = {
            'image_name': user_metadata.get('image_name', 'ISIC_6724629'),
            'patient_id': user_metadata.get('patient_id', 'IP_9738076'),
            'sex': user_metadata.get('sex', 'male'),
            'age_approx': user_metadata.get('age_approx', 50),
            'anatom_site_general_challenge': user_metadata.get('anatom_site_general_challenge', 'head/neck'),
            'width': 1920,
            'height': 1080
        }

        new_data_dict = {
            'image_name': data_dict['image_name'],
            'patient_id': data_dict['patient_id'],
            'sex': 1 if data_dict['sex'] == 'male' else 0,  # Assuming 1 for female, 0 for other
            'age_approx': data_dict['age_approx'] / 100,  # Normalize age between 0 and 1
            'anatom_site_general_challenge': data_dict['anatom_site_general_challenge'],
            'width': data_dict['width'],
            'height': data_dict['height'],
            'site_head/neck': 1 if data_dict['anatom_site_general_challenge'] == 'head/neck' else 0,
            'site_lower extremity': 1 if data_dict['anatom_site_general_challenge'] == 'lower extremity' else 0,
            'site_oral/genital': 1 if data_dict['anatom_site_general_challenge'] == 'oral/genital' else 0,
            'site_palms/soles': 1 if data_dict['anatom_site_general_challenge'] == 'palms/soles' else 0,
            'site_torso': 1 if data_dict['anatom_site_general_challenge'] == 'torso' else 0,
            'site_upper extremity': 1 if data_dict['anatom_site_general_challenge'] == 'upper extremity' else 0,
            'site_nan': 1 if pd.isna(data_dict['anatom_site_general_challenge']) else 0,
        }

        # print(new_data_dict)
        test_df = pd.DataFrame([new_data_dict])


        meta_features = ['sex', 'age_approx', 'site_head/neck', 'site_lower extremity', 'site_oral/genital', 'site_palms/soles', 'site_torso', 'site_upper extremity', 'site_nan']

        #['sex', 'age_approx'] + [col for col in train_df.columns if 'site_' in col]
        #meta_features.remove('anatom_site_general_challenge')

        test = MelanomaDataset(df=test_df,
                            file_path=file_path, 
                            train=False,
                            transforms=train_transform,  # For TTA
                            meta_features=meta_features)

        skf = GroupKFold(n_splits=5)

        epochs = 12  # Number of epochs to run
        es_patience = 3  # Early Stopping patience - for how many epochs with no improvements to wait
        TTA = 3 # Test Time Augmentation rounds

        # oof = np.zeros((len(train_df), 1))  # Out Of Fold predictions
        preds = torch.zeros((len(test), 1), dtype=torch.float32, device=device)  # Predictions for test test

        skf = KFold(n_splits=5, shuffle=True, random_state=47)

        arch = EfficientNet.from_pretrained('efficientnet-b1')
        model = Net(arch=arch, n_meta_features=len(meta_features))  # New model for each fold
        test_loader = DataLoader(dataset=test, batch_size=1, shuffle=False, num_workers=0)

        def predict_test_set(model, test_loader, device, TTA):
            model.eval()  # Set the model to evaluation mode
            predictions = torch.zeros((len(test_loader.dataset), 1), dtype=torch.float32, device=device)

            with torch.no_grad():  # Disable gradient calculation during inference
                for _ in range(TTA):
                    for i, x_test in enumerate(test_loader):
                        x_test[0] = torch.tensor(x_test[0], device=device, dtype=torch.float32)
                        x_test[1] = torch.tensor(x_test[1], device=device, dtype=torch.float32)

                        # Forward pass
                        z_test = model(x_test)
                        #print(z_test)

                        pred_test = torch.sigmoid(z_test)

                        # Update predictions
                        predictions[i * test_loader.batch_size:i * test_loader.batch_size + x_test[0].shape[0]] += pred_test

            # Average predictions over TTA
            predictions /= TTA
            return predictions.cpu().numpy()


        def return_pred(device=device):
            preds_sum = 0  # Variable to accumulate predictions
            current_dir = os.getcwd()
            for i in range(1, 6):
                import __main__
                setattr(__main__, "Net", Net)
                
                model = Net(arch=arch, n_meta_features=len(meta_features))
                model = torch.load(f'{current_dir}/model_{i}.pth')
                model.to(device)
                # model.eval()

                preds_i = predict_test_set(model=model, test_loader=test_loader, device=device, TTA=5)
                preds_i = preds_i.reshape(-1)  # Reshape the predictions

            
                preds_sum += preds_i
            return preds_sum / 5  # Average predictions over 5 models

        return return_pred()