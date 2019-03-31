from torchvision import models, transforms
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn

from explainer.gradcam import *
from explainer.backprop import *
from explainer.deeplift import *
from explainer.occlusion import *
from  explainer.epsilon_lrp import *



from skimage.transform import resize


def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])       
    transf = transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor(),
        normalize
    ])    

    return transf

def get_model(model_name):
    model = models.__dict__[model_name](pretrained=True)
    return model

def get_image(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def img2tensor(image):
    transf = get_input_transform()
    image = transf(image)
    return Variable(image.unsqueeze(0), requires_grad=False)

def class2tensor(target_class):
    return torch.LongTensor([target_class])

def show_saliency(saliency, input, cmap='jet', alpha=0.6):
    plt.imshow(input)
    plt.imshow(saliency, alpha=alpha, cmap=cmap)
    plt.show()

def get_explainer(explainer_name, model, layer_path=None):
    if explainer_name == 'gradcam':
        return GradCAMExplainer(model, target_layer_name_keys=layer_path, use_inp=True)
    if explainer_name == 'vanilla_grad':
        return VanillaGradExplainer(model)
    if explainer_name == 'grad_x_input':
        return GradxInputExplainer(model)
    if explainer_name == 'saliency':
        return SaliencyExplainer(model)
    if explainer_name == 'integrate_grad':
        return IntegrateGradExplainer(model)
    if explainer_name == 'deconv':
        return DeconvExplainer(model)
    if explainer_name == 'guided_backprop':
        return GuidedBackpropExplainer(model)
    if explainer_name == 'smooth_grad':
        return SmoothGradExplainer(model)
    if explainer_name == 'deeplift_rescale':
        return DeepLIFTRescaleExplainer(model)
    if explainer_name == 'occlusion':
        return OcclusionExplainer(model)
    if explainer_name == 'lrp':
        return EpsilonLrp(model)

    raise ValueError('Explainer {} is not recognized'.format(explainer_name))

def get_input():
    raw_input = get_image('images/elephant.png')
    input = img2tensor(raw_input)
    target_class = class2tensor(101)
    return raw_input, input, target_class

def main():
    model = get_model('vgg16')
    exp = get_explainer('lrp', model, ['avgpool'])
    #exp = get_explainer('gradcam', model, ['avgpool'])
    raw_input, input, target_class = get_input()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input = input.to(device)
    target_class = target_class.to(device)

    saliency = exp.explain(input, target_class)

    saliency = saliency.abs().sum(dim=1)[0].squeeze()
    saliency -= saliency.min()
    saliency /= (saliency.max() + 1e-20)
    #upsampler = nn.Upsample(size=(raw_input.height, raw_input.width), mode='bilinear')
    saliency_upsampled = resize(saliency.detach().cpu().numpy(), (raw_input.height, raw_input.width))

    show_saliency(saliency_upsampled, raw_input)

main()