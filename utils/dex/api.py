import os

import torch
import torchvision.transforms as transforms

from .models import Age, Gender

device = 'cuda'

age_model = Age()
gender_model = Gender()

cwd = os.path.dirname(__file__)
age_model_path = os.path.join(cwd, 'pth/age_sd.pth')
gender_model_path = os.path.join(cwd, 'pth/gender_sd.pth')


def _eval():
    global age_model
    global gender_model
    age_model.load_state_dict(torch.load(age_model_path))
    age_model.eval()
    age_model = age_model.to(device)
    gender_model.load_state_dict(torch.load(gender_model_path))
    gender_model.eval()
    gender_model = gender_model.to(device)


def expected_age(tensor):
    weight = torch.arange(1, 102, device=device)
    return weight * tensor


def estimate_age(img):
#    tensor = transforms.CenterCrop(224)(img)
#     tensor = transforms.CenterCrop(224)(img)
    h = img.size(2)
    offset = (h - 224) // 2
    tensor = img[:, :, offset:-offset, offset:-offset]
#     print(tensor.shape)

    with torch.no_grad():
        output = age_model(tensor)
    age = expected_age(output)
    return torch.sum(age, dim=1)


def estimate_gender(img):
    tensor = transforms.CenterCrop(224)(img)
    with torch.no_grad():
        output = gender_model(tensor)
    return output
