import torch
import torchvision



model = torchvision.models.resnet18(pretrained=True)
script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')