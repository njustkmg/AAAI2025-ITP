import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
import numpy as np

class resnet50_modified(nn.Module):
    def __init__(self):
        super(resnet50_modified, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = None
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gradients = []
        self.activations = []
        self.handles_list = []
        self.pruned_activations_mask = []
    
    def forward_features(self, x, threshold=1e6):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.clip(max = threshold)
        x = x.view(x.size(0), -1)  
        return x
    
    def forward_head(self, x):
        x = self.model.fc(x)
        return x
    
    def forward_threshold(self, x, threshold=1e6):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.clip(max = threshold)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)  
        return x
    
    def _forward(self, x, fc_params=None):
        self.activations = []
        self.gradients = []
        self.zero_grad()
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)  
        return x
    
    def remove_handles(self):
        for handle in self.handles_list:
            handle.remove()
        self.handles_list.clear()
        self.activations = []
        self.gradients = []

    def _compute_taylor_scores(self, inputs, labels):
        self._hook_layers()
        outputs = self._forward(inputs)
        outputs[0, labels.item()].backward(retain_graph=True)

        first_order_taylor_scores = []
        self.gradients.reverse()

        for i, layer in enumerate(self.activations):
            first_order_taylor_scores.append(torch.mul(layer, self.gradients[i]))
        
        self.remove_handles()                
        return first_order_taylor_scores
    
    def _hook_layers(self):
        def backward_hook_relu(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].to(self.device))

        def forward_hook_relu(module, input, output):
            if self.pruned_activations_mask:
              output = torch.mul(output, self.pruned_activations_mask[len(self.activations)].to(self.device)) #+ self.pruning_biases[len(self.activations)].to(self.device)
            self.activations.append(output.to(self.device))
            return output

        for module in self.modules():
            if isinstance(module, nn.AdaptiveAvgPool2d):
                self.handles_list.append(module.register_forward_hook(forward_hook_relu))
                self.handles_list.append(module.register_backward_hook(backward_hook_relu))