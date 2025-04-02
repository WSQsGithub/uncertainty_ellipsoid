import torch
import torch.nn as nn
from typing import Literal, Optional


class UncertaintyEllipsoidLoss(nn.Module):
    def __init__(self, task: Literal["train_center", "train_shape", "train_end2end"], lambda_center=1.0, lambda_containment=0.5, lambda_reg=0.1):
        """
        Initialize the Uncertainty Ellipsoid Loss function.

        Args:
            task (str): Task type, either "train_center" or "train_shape" or "train_end2end"
            lambda_center (float): Hyperparameter controlling the center loss weight
            lambda_containment (float): Hyperparameter controlling the containment loss weight
            lambda_reg (float): Hyperparameter controlling the regularization loss weight
        """
        super(UncertaintyEllipsoidLoss, self).__init__()
        self.task = task
        self.lambda_center = lambda_center
        self.lambda_containment = lambda_containment
        self.lambda_reg = lambda_reg

    def center_loss(self, world_coords, pred_center):
        """
        Center loss: Mean squared error between the predicted center and true center.

        Args:
            world_coords (Tensor): World coordinates of shape (N, M_S, 3)
            pred_center (Tensor): Predicted center coordinates of shape (N, 3)

        Returns:
            Tensor: Center loss value.
        """
        true_center = world_coords.mean(
            dim=1
        )  # True center is the mean of world coordinates along M_S dimension
        loss = torch.mean((true_center - pred_center) ** 2)
        return loss

    def containment_loss(self, world_coords, pred_center, P: Optional[torch.Tensor] = None):
        """
        Containment loss: Mean distance between the true world coordinates outside the ellipsoid and the ellipsoid surface.

        Args:
            world_coords (Tensor): World coordinates of shape (N, M_S, 3)
            pred_center (Tensor): Predicted center coordinates of shape (N, 3)
            P (Tensor): Positive definite matrix, shape (N, 3, 3)

        Returns:
            Tensor: Containment loss value.
        """
        if P is None:
            return 0

        N, M_S, _ = world_coords.size()
        diff = world_coords - pred_center.unsqueeze(1)  # Shape (N, M_S, 3)

        # Calculate (x_ij - c_i)^T * P_i * (x_ij - c_i)
        distances = torch.bmm(diff, P)  # Shape (N, M_S, 3)
        distances = torch.bmm(distances, diff.transpose(1, 2))  # Shape (N, M_S, M_S)
        distances = torch.diagonal(distances, dim1=1, dim2=2) - 1  # Shape (N,M_S)

        # Containment loss: max(0, (x_ij - c_i)^T P_i (x_ij - c_i) - 1)
        containment_losses = torch.mean(torch.sigmoid(distances * 1000), dim=1)  # Shape (N,)

        # Average the loss across all samples
        containment_loss = containment_losses.mean()  # Scalar
        return containment_loss

    def regularization_loss(self, L=None):
        """
        Regularization loss: Ensures the ellipsoid is not too large by minimizing the trace of P.

        Args:
            L (Tensor): Lower triangular matrix from the Cholesky decomposition, shape (N, 3, 3)

        Returns:
            Tensor: Regularization loss value.
        """
        if L is None:
            return 0
        # Regularization is the trace of P = L^T * L
        # Since trace(P) = trace(L^T * L) = sum(diagonal(L^T * L)) = sum(diagonal(L * L^T))
        l11, _, l22, _, _, l33 = (
            L[:, 0, 0],
            L[:, 1, 0],
            L[:, 1, 1],
            L[:, 2, 0],
            L[:, 2, 1],
            L[:, 2, 2],
        )
        det_L = l11 * l22 * l33

        reg_loss = 1 / torch.abs(det_L)

        return reg_loss.mean()

    def volume_loss(self, world_coords, L: Optional[torch.Tensor] = None):
        """
        Volume loss: the volume difference between the predicted ellipse and convex hull

        Args:
            world_coords (Tensor): World coordinates of shape (N, M_S, 3)
            L (Tensor): Lower triangular matrix from the Cholesky decomposition, shape (N, 3, 3)

        Return:
            Tensor: The volume loss
        """
        if L is None:
            return 0
        l11, l21, l22, l31, l32, l33 = (
            L[:, 0, 0],
            L[:, 1, 0],
            L[:, 1, 1],
            L[:, 2, 0],
            L[:, 2, 1],
            L[:, 2, 2],
        )
        det_L = l11 * l22 * l33

        vol_predicted = 4 / 3 * torch.pi * torch.abs(det_L)

        vol_hull = None  # TODO: finish this value

        vol_diff = vol_predicted - vol_hull

        return torch.sigmoid(1000 * vol_diff).mean()  # to enlarge the difference

    def forward(self, world_coords, pred_center, L: Optional[torch.Tensor] = None):
        """
        Compute the total loss.

        Args:
            world_coords (Tensor): World coordinates of shape (N, M_S, 3)
            pred_center (Tensor): Predicted center coordinates of shape (N, 3)
            L (Tensor): Lower triangular matrix from the Cholesky decomposition, shape (N, 3, 3)

        Returns:
            Tensor: The total loss value.
        """
        # Compute P once in the forward function
        P = torch.bmm(L.transpose(1, 2), L) if L else None  # Shape (N, 3, 3)

        # Compute the three components of the loss
        loss_center = self.center_loss(world_coords, pred_center)
        loss_containment = self.containment_loss(world_coords, pred_center, P)
        loss_reg = self.regularization_loss(L)

        # Combine them with their respective lambda weights
        total_loss = (
            self.lambda_center * loss_center
            + self.lambda_containment * loss_containment
            + self.lambda_reg * loss_reg
        )

        info = {
            "lambda": {
                "center": self.lambda_center,
                "containment": self.lambda_containment,
                "regularization": self.lambda_reg,
            },
            "loss": {
                "total": total_loss,
                "center": loss_center,
                "containment": loss_containment,
                "regularization": loss_reg,
            },
        }

        return total_loss, info


# Example usage:
# Assuming you have `world_coords`, `pred_center`, and `L` tensors available
# world_coords: Tensor of shape (N, M_S, 3)
# pred_center: Tensor of shape (N, 3)
# L: Tensor of shape (N, 3, 3) (lower triangular matrix from Cholesky decomposition)
if __name__ == "__main__":
    loss_fn = UncertaintyEllipsoidLoss(lambda_center=1.0, lambda_containment=1.0, lambda_reg=1.0)

    # Example tensors (random for illustration)
    N = 5  # Batch size
    M_S = 10  # Number of samples per batch
    world_coords = torch.randn(N, M_S, 3)
    pred_center = torch.randn(N, 3)
    L = torch.randn(N, 3, 3).tril()  # Ensure L is lower triangular

    # Compute the loss
    loss, info = loss_fn(world_coords, pred_center, L)
    print("Info:", info)
