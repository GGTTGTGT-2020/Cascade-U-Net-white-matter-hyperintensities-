# Cascade-U-Net-white-matter-hyperintensities
Pytorch implementation of Cascade U-Net for pvWMHs/dWMHs segmentation. The well-trained model was uploaded in the release page. Notice that this model was trained on a community-based dataset, fine-tuning is suggested before usage. Z-score normalization should be conducted to the data before testing.

### Usage 

import torch

from Model import Cascade_UNet

### Define model
model = Cascade_UNet(n_channels=1, n_classes=3)

### Define the device
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu') 

model.to(device = device)

### Load weights
weights = './Cascade_U.pth'

model.load_state_dict(torch.load(weights, map_location = device))

### For further training:
optimizer = torch.optim.Adam(model.parameters(), lr = 0.002)

loss = nn.CrossEntropyLoss()

model.train()

...

optimizer.zero_grad()

loss.backward()

optimizer.step()

### For testing, load the ‘data’ as tensor:
data = data.to(device = device)

model.eval()

pred = model(data)

probs = torch.nn.functional.softmax(pred, dim = 1)
