import torch
import torch.nn as nn

class CNNRegressionModel(nn.Module):
    """
    CNN Regression Model for steering angle prediction.

    Architecture inspired by NVIDIA's DAVE-2 self-driving CNN (2016),
    adapted for the 66x200 synthetic driving images used in this project.

    Input  : (batch, 3, 66, 200)  — RGB image
    Output : (batch, 1)           — predicted steering angle

    Structure:
      Block 1-5  : Convolutional layers (feature extraction)
                   Each block = Conv2d → BatchNorm → ReLU → Dropout2d
      Flatten
      FC 1-3     : Fully-connected regression head
                   Each = Linear → ReLU → Dropout
      Output     : Linear(64 → 1)  — single steering value

    BatchNorm  : stabilises training on small/synthetic datasets
    Dropout    : reduces overfitting (p=0.2 conv, p=0.3 fc)
    """

    def __init__(self, dropout_conv: float = 0.2, dropout_fc: float = 0.3):
        super().__init__()

        # Convolutional feature extractor 
        self.conv_block = nn.Sequential(

            # Block 1 : (3, 66, 200) → (24, 31, 98)
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_conv),

            # Block 2 : (24, 31, 98) → (36, 14, 47)
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_conv),

            # Block 3 : (36, 14, 47) → (48, 5, 22)
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_conv),

            # Block 4 : (48, 5, 22) → (64, 3, 20)
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_conv),

            # Block 5 : (64, 3, 20) → (64, 1, 18)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Automatically compute flattened feature size 
        self._flat_size = self._get_flat_size()

        # Fully-connected regression head 
        self.fc_block = nn.Sequential(
            nn.Linear(self._flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 1),          # final steering output
        )

        # Weight initialisation (He for ReLU networks) 
        self._init_weights()

    def _get_flat_size(self) -> int:
        """Pass a dummy tensor through conv_block to get its output size."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 66, 200)
            out   = self.conv_block(dummy)
            return int(out.view(1, -1).shape[1])

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x : (batch, 3, 66, 200)  — normalised float tensor
        Returns:
          (batch, 1)               — predicted steering angle
        """
        features = self.conv_block(x)
        features = features.view(features.size(0), -1)   # flatten
        return self.fc_block(features)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)