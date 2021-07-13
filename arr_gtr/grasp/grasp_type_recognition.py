# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import PIL
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from arr_gtr.data.grasp import get_gtr_data
from arr_gtr.data.grasp import get_uniformal_affordance
from arr_gtr.data.grasp import get_varied_affordance
from arr_gtr.grasp.grasp_type_utils import grasp_types


class GraspTypeRecognitionModule(nn.Module):

    def __init__(self, num_class=None, pretrained_model=None):
        super(GraspTypeRecognitionModule, self).__init__()
        if num_class is None:
            pass
        model = models.resnet101(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features
        num_class = len(grasp_types)
        model.fc = nn.Linear(num_features, num_class)

        if pretrained_model is None:
            pretrained_model = get_gtr_data()
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(pretrained_model))
        else:
            model.load_state_dict(torch.load(pretrained_model, map_location=torch.device('cpu')))
        self.model = model

        self.transform = transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])

        self.uniformal_affordance = get_uniformal_affordance()
        self.varied_affordance = get_varied_affordance()

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)

    def inference(self, img):
        """Inference single image.

        Parameters
        ----------
        img : numpy.ndarray or PIL.Image.Image
            single image.
        """
        if isinstance(img, np.ndarray):
            img = PIL.Image.fromarray(img)
        if not isinstance(img, PIL.Image.Image):
            raise TypeError
        tensor_img = self.transform(img)
        device = next(self.parameters()).device
        tensor_img = tensor_img.to(device)
        return self.forward(tensor_img[None, ]).to("cpu").detach().numpy()[0]

    def inference_with_affordance(
            self, img, object_name, affordance_type='varied'):
        """Inference single image.

        Parameters
        ----------
        img : numpy.ndarray or PIL.Image.Image
            single image.
        object_name : str
            target object name.
        affordance_type : str
            ['varied', 'uniformal']
        """
        if affordance_type == 'varied':
            pgo = self.varied_affordance[object_name]
        elif affordance_type == 'uniformal':
            pgo = self.uniformal_affordance[object_name]
        else:
            raise ValueError
        pgi = self.inference(img)
        pg = pgi * pgo
        return pg

    def inference_from_affordance(self, object_name, affordance_type='varied'):
        """Inference only using affordance.

        Parameters
        ----------
        object_name : str
            target object name.
        affordance_type : str
            ['varied', 'uniformal']
        """
        if affordance_type == 'varied':
            pgo = self.varied_affordance[object_name]
        elif affordance_type == 'uniformal':
            pgo = self.uniformal_affordance[object_name]
        else:
            raise ValueError
        return pgo
