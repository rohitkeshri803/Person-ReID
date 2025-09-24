import torch.nn as nn
import torchvision
from data_loader import load_model
import torch as T

class model(nn.Module):
    '''
    input: N * 3 * 224 * 224
    output: N * out_dim, N * inter_dim, N * C' * 7 * 7
    '''
    def __init__(self, freeze_param=False, inter_dim=512, out_dim=256, model_path=None):
        super(model, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        state_dict = self.backbone.state_dict()
        num_features = self.backbone.fc.in_features
        #print(f"num_features: { num_features }")
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        model_dict = self.backbone.state_dict()
        model_dict.update({k: v for k, v in state_dict.items() if k in model_dict})
        self.backbone.load_state_dict(model_dict)
        if freeze_param:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.avg_pooling = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(num_features, inter_dim)
        self.fc2 = nn.Linear(inter_dim, out_dim)
        state = load_model(model_path)
        if state:
            new_state = self.state_dict()
            new_state.update({k: v for k, v in state.items() if k in new_state})
            self.load_state_dict(new_state)

    def forward(self, x):
        x = self.backbone(x)
        pooled = self.avg_pooling(x)
        #inter_out = self.fc(pooled.view(pooled.size(0), -1))
        #out = self.fc2(inter_out)
        pooled = T.squeeze(pooled)
        if len(list(pooled.shape)) == 1: #reshape from (2048, ) to (1, 2048)
          pooled = pooled.reshape(1, -1)
        #print(f"pooled: { pooled.shape }")
        return pooled