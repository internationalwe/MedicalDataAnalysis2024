import albumentations as A
from torchvision import transforms

def get_transform_function(transform_function_str, config):
    if transform_function_str == "baseTransform":
        return baseTransform(config)
    elif transform_function_str == "medicalTransform":
        return medicalTransform(config)
    elif transform_function_str == "rotate_affine_crop_horizontal_colorjitter":
        return rotate_affine_crop_horizontal_colorjitter(config)
    elif transform_function_str == "only_colorjitter":
        return only_colorjitter(config)


def baseTransform(config):
    return A.Compose(
        [
            A.Resize(config["input_width"], config["input_height"]),
        ]
    )


def medicalTransform(config):
    return A.Compose(
        [
            # 이미지를 임의의 크기로 자르고 다시 조정합니다. 50%의 확률로 적용됩니다.
            A.RandomResizedCrop(config["input_size"], config["input_size"], p=0.5),
            # 이미지에 탄성 변형을 적용합니다. 50%의 확률로 적용됩니다.
            A.ElasticTransform(p=0.5),
            # 이미지를 압축하여 품질을 조정합니다. 50%의 확률로 적용됩니다.
            A.ImageCompression(quality_lower=85, quality_upper=95, p=0.5),
            # 이미지의 밝기와 대비를 무작위로 조정합니다. 50%의 확률로 적용됩니다.
            A.RandomBrightnessContrast(p=0.5),
            # 블러(Blur) 또는 중앙값 블러(MedianBlur) 중 하나를 무작위로 선택하여 적용합니다. 50%의 확률로 적용됩니다.
            A.OneOf([A.Blur(blur_limit=3), A.MedianBlur(blur_limit=3)], p=0.5),
            # 이미지를 수평으로 뒤집습니다. 50%의 확률로 적용됩니다.
            A.HorizontalFlip(p=1),
            # 이미지의 대비를 제한된 지역에서 증가시킵니다. 50%의 확률로 적용됩니다.
            A.CLAHE(p=1),
            # 이미지의 선명도를 높입니다. 50%의 확률로 적용됩니다.
            A.Sharpen(p=0.5),
            # 이미지를 밝게 만듭니다. 50%의 확률로 적용됩니다.
            A.Brightness(p=0.5),
        ]
    )

def rotate_affine_crop_horizontal_colorjitter(config):
    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=(-90, 90)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 조절 (잡음 추가)
            transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 무작위 변환 (이동)
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 무작위 줌인
            transforms.RandomHorizontalFlip(p = 1),  # 무작위로 수평으로 뒤집기
            
            transforms.Resize((config["input_width"], config["input_height"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

def only_colorjitter(config):
    return transforms.Compose(
        [
            transforms.ColorJitter(brightness=(0.5, 0.9), 
                        contrast=(0.4, 0.8), 
                        saturation=(0.7, 0.9),
                        hue=(-0.2, 0.2),
            ),
            transforms.Resize((config["input_width"], config["input_height"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

def only_colorjitter2(config):
    return transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 조절 (잡음 추가)
            transforms.Resize(config["input_width"], config["input_height"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

def rotate_affine_crop_horizontal_colorjitter2(config):
    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=(-90, 90)),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 무작위 변환 (이동)
            transforms.RandomHorizontalFlip(p = 1),  # 무작위로 수평으로 뒤집기
            transforms.ColorJitter(brightness=(0.5, 0.9), 
                                contrast=(0.4, 0.8), 
                                saturation=(0.7, 0.9),
                                hue=(-0.2, 0.2),
                                ),
            transforms.Resize(config["input_width"], config["input_height"]), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )