import torch
from torch.utils.data import DataLoader
from dataset.dataset_loader import DrivingDataset
from models.steering_model import SteeringRegression

def train_model(epochs=5):

    dataset = DrivingDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SteeringRegression()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for images, angles in loader:
            outputs = model(images)
            loss = criterion(outputs, angles)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")

    return model