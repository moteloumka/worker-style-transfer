
import sys
import os
root_dir = os.path.dirname(__file__)
# Add the project directory to the Python path
sys.path.append(os.path.join(root_dir,'Collaborative-Distillation', 'PytorchWCT'))
root_dir = os.path.abspath(os.path.join(os.path.join('Collaborative-Distillation')))
import torch
from PIL import Image
from util_wct import WCT
import base64
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
import base64
from data_loader import SingleImageDataset
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import runpod

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

ALPHA = 0.5

# relative path in Collaborative-Distillation directory
def get_absolute_path(relative_path):
    return os.path.join(root_dir, relative_path)

def decode_base64_to_tensor(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return transforms.ToTensor()(image)

# also normalizes the image
def get_img_from_tensor(cImg):
    cImg = cImg.squeeze(0)
    cImg = cImg / cImg.max()
    cImg = cImg.type(torch.float32)
    to_pil = ToPILImage()
    image = to_pil(cImg)
    return image

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def get_model_paths():
    paths = {}
    paths['e5'] = get_absolute_path('trained_models/wct_se_16x_new/5SE.pth')
    paths['e4'] = get_absolute_path('trained_models/wct_se_16x_new/4SE.pth')
    paths['e3'] = get_absolute_path('trained_models/wct_se_16x_new/3SE.pth')
    paths['e2'] = get_absolute_path('trained_models/wct_se_16x_new/2SE.pth')
    paths['e1'] = get_absolute_path('trained_models/wct_se_16x_new/1SE.pth')
    paths['d5'] = get_absolute_path('trained_models/wct_se_16x_new_sd/5SD.pth')
    paths['d4'] = get_absolute_path('trained_models/wct_se_16x_new_sd/4SD.pth')
    paths['d3'] = get_absolute_path('trained_models/wct_se_16x_new_sd/3SD.pth')
    paths['d2'] = get_absolute_path('trained_models/wct_se_16x_new_sd/2SD.pth')
    paths['d1'] = get_absolute_path('trained_models/wct_se_16x_new_sd/1SD.pth')
    return paths

@torch.no_grad()
def styleTransfer(encoder, decoder, contentImg, styleImg, csF, wct):
    sF  = encoder(styleImg); torch.cuda.empty_cache() # empty cache to save memory
    cF  = encoder(contentImg); torch.cuda.empty_cache()
    sF  = sF.data.cpu().squeeze(0) # note: svd runs on CPU
    cF  = cF.data.cpu().squeeze(0)
    csF = wct.transform(cF, sF, csF,ALPHA)
    Img = decoder(csF); torch.cuda.empty_cache()
    return Img


def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    content_encoded = job_input['content']
    style_encoded = job_input['style']

    content_tensor = decode_base64_to_tensor(content_encoded)
    style_tensor = decode_base64_to_tensor(style_encoded)

    dataset = SingleImageDataset(content_tensor, style_tensor)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    paths = get_model_paths()
    wct = WCT(paths).cuda()
    csF = torch.Tensor().cuda()
    NB_RUN_WCT = 1 # number of WCT passes, can be >= 1, but high values yield strange results

    for _ , (cImg, sImg) in enumerate(loader):
        cImg, sImg = cImg.squeeze(0), sImg.squeeze(0)
        cImg = cImg.cuda()
        sImg = sImg.cuda()

        for k in range(NB_RUN_WCT):
          cImg = styleTransfer(wct.e5, wct.d5, cImg, sImg, csF, wct)
          cImg = styleTransfer(wct.e4, wct.d4, cImg, sImg, csF, wct)
          cImg = styleTransfer(wct.e3, wct.d3, cImg, sImg, csF, wct)
          cImg = styleTransfer(wct.e2, wct.d2, cImg, sImg, csF, wct)
          cImg = styleTransfer(wct.e1, wct.d1, cImg, sImg, csF, wct)

        cImg = cImg.data.cpu()
        
    image = get_img_from_tensor(cImg)
    encoding = image_to_base64(image)

    result = {"image": encoding}
    return result
    


runpod.serverless.start({"handler": handler})
