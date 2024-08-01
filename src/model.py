import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()
        
        # (3x224x224) RGB images input
        # Define the convolutional layers as a Sequential block
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), #(16*224*224)
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2), #(16*112*112)
            nn.ReLU(inplace=True),
           
            nn.Conv2d(16, 32, 3, padding=1), #(32*112*112)
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2), #(32*56*56)
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1), #(64*56*56)
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2), #(64*28*28)
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, padding=1), #(128*28*28)
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2), #(128*14*14)
            nn.ReLU(inplace=True),

            
        )

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 14 * 14, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)

        # Flatten batch_size
        x = x.view(x.size(0), -1) # Dynamically compute the flattened size

        x = self.dropout(x)
        x = self.fc1(x)

        x = torch.relu(self.fc1_bn(x))

        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(self.fc2_bn(x))

        x = self.dropout(x)
        x = self.fc3(x)

        return x



######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
