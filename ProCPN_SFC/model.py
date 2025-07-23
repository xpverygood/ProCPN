
import torch
import torch.nn as nn
from helpers import SinusoidalPosEmb # Use absolute import

class MLP(nn.Module):
    """
    Conditional MLP for the Diffusion model.
    Takes noisy action parameters (x), timestep (time), and environment state (state) as input.
    Includes Layer Normalization for potentially improved stability.
    """
    def __init__(
        self,
        state_dim: int,      # Dimension of the flattened observation vector from SFCEnv
        action_dim: int,     # Dimension of the action parameters being diffused (e.g., logits)
        hidden_dim: int = 256,
        t_dim: int = 16,
        activation: str = 'relu' # Assuming ReLU based on previous steps
    ):
        super(MLP, self).__init__()
        _act = nn.ReLU # Use ReLU

        # --- State MLP with LayerNorm ---
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # <--- Added LayerNorm
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # <--- Added LayerNorm
            _act(),
            nn.Linear(hidden_dim, hidden_dim) # Output dimension matches hidden_dim for merging
        )

        # --- Time MLP ---
        # LayerNorm is optional here, often less critical than for state/combined features
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, hidden_dim)
            # nn.LayerNorm(hidden_dim), # Optional Norm
        )

        # --- Action Input Layer ---
        # LayerNorm is optional here
        self.action_layer = nn.Sequential(
             nn.Linear(action_dim, hidden_dim),
             # nn.LayerNorm(hidden_dim), # Optional Norm
             _act()
        )


        # --- Mid Layer with LayerNorm ---
        # Combines processed state, time, and action information
        self.mid_layer = nn.Sequential(
            # Input dim: hidden_dim (action) + hidden_dim (time) + hidden_dim (state)
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2), # <--- Added LayerNorm
            _act(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim), # <--- Added LayerNorm
            _act(),
            nn.Linear(hidden_dim, action_dim) # Output the prediction (noise or x0)
        )

        # --- Final Layer ---
        self.final_layer = nn.Identity()

    def forward(self, x, time, state):
        """
        Args:
            x (torch.Tensor): Noisy action parameters (batch_size, action_dim)
            time (torch.Tensor): Timestep (batch_size,)
            state (torch.Tensor): Flattened environment observation (batch_size, state_dim)

        Returns:
            torch.Tensor: Predicted noise or x0 (batch_size, action_dim)
        """
        processed_state = self.state_mlp(state)
        processed_time = self.time_mlp(time)
        processed_action = self.action_layer(x)

        # Concatenate the processed embeddings
        combined = torch.cat([processed_action, processed_time, processed_state], dim=1)

        # Pass through the middle layers to get the final prediction
        output = self.mid_layer(combined)
        output = self.final_layer(output)

        # --- Optional check within MLP forward ---
        # if torch.isnan(output).any() or torch.isinf(output).any():
        #     print("!!! NaN/Inf detected inside MLP forward output !!!")
        #     output = torch.nan_to_num(output) # Apply nan_to_num here if needed as last resort
        # --- End Optional check ---

        return output


class ValueCritic(nn.Module):
    """
    Critic network that estimates the state value V(s).
    Takes the flattened environment state (observation) as input.
    Includes Layer Normalization for potentially improved stability.
    """
    def __init__(
            self,
            state_dim: int,       # Dimension of the flattened observation vector from SFCEnv
            hidden_dim: int = 256,
            activation: str = 'relu' # Use ReLU
    ):
        super(ValueCritic, self).__init__()
        _act = nn.ReLU

        # --- State MLP with LayerNorm ---
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # <--- Added LayerNorm
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # <--- Added LayerNorm
            _act()
        )

        # --- Value Head ---
        # LayerNorm less common right before a single scalar output
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            _act(),
            nn.Linear(hidden_dim // 2, 1) # Final output layer
        )

    def forward(self, state):
        """
        Args:
            state (torch.Tensor): Flattened environment observation (batch_size, state_dim)

        Returns:
            torch.Tensor: Estimated state value V(s) (batch_size, 1)
        """
        processed_state = self.state_mlp(state)
        value = self.value_head(processed_state)
        return value
