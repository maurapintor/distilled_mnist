class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 200)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = torch.relu(x)

        return self.linear3(x)

def load_model(weigths_url: str):
    model = MNISTModel()
    path = Path("models/mnist_distilled.pt")
    if not path.exists():
        download_gdrive(MODEL_ID, path)
    state_dict = torch.load(
        path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model