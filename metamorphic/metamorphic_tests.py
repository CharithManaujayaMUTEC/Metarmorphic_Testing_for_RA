import torch
import torchvision.transforms.functional as TF


def horizontal_flip_test(model, image, steering):

    flipped_image = TF.hflip(image)
    flipped_prediction = model(flipped_image.unsqueeze(0))

    original_prediction = model(image.unsqueeze(0))

    # Steering should invert sign
    expected = -original_prediction

    difference = torch.abs(flipped_prediction - expected)

    return difference.item()