import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
import torch.nn.functional as F
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization,ReLU, Add, MaxPool2D, GlobalAvgPool2D, Dense
# in-repo imports
from mymodel import t2d
from myloader import CDFL
from myrealmodel import Net
if __name__ == "__main__":
    transformations = transforms.Compose([transforms.ToTensor()])
    custom_mnist_from_csv_loc =  CDFL()

    dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_csv_loc,
                                                    batch_size=10,
                                                    shuffle=False)
    #model = Net()
    model=t2d(10)
    print(model)
    
    #criterion=F.nll_loss()
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i, (images, labels) in enumerate(dataset_loader):
        #print(images[0])
        images = Variable(images)
        labels = Variable(labels)
        # Clear gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        #print(outputs)
        # Calculate loss
        #print(labels)
        #loss = criterion(outputs, labels)
        loss=F.nll_loss(outputs,labels)   
        # Backward pass
        print(loss)
        loss.backward()
        # Up    # Preprocessing operations are defined inside the datasetdate weights
        optimizer.step()
        #break

