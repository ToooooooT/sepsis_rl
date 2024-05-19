import torch
import torch.optim as optim
from torch.distributions import Categorical, kl_divergence

MAX_KL_DIV = 20.0

class BC:
    def __init__(
        self,
        device: torch.device,
        bc_type: str,
        bc_kl_beta: float,
        use_pi_b_kl: bool,
        nu_lr: float,
        pi_b_model_path: str = 'pi_b_models/model_mean_agg.pth',
    ):
        self.bc_type = bc_type
        self.nu_lr = nu_lr
        if self.bc_type == "KL":
            self.log_nu = torch.zeros(1, dtype=torch.float, device=device, requires_grad=True)
            self.nu_optimizer = optim.Adam([self.log_nu], lr=self.nu_lr, eps=1e-4)
            self.bc_kl_beta = bc_kl_beta
            self.use_pi_b_kl = use_pi_b_kl
            if self.use_pi_b_kl:
                self.pi_b_model = torch.load(pi_b_model_path, map_location=torch.device('cpu')).to(device)
                self.pi_b_model.eval()

    def get_behavior(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        action_probs: torch.Tensor,
        device: torch.device,
    ) -> Categorical:
        if self.use_pi_b_kl:
            behavior_logits = self.pi_b_model(states)
            behavior = Categorical(logits=behavior_logits)
        else:
            # assume other action probabilities is 0.01 of behavior policy
            epsilon = 0.01
            behavior_probs = torch.full(action_probs.shape, epsilon, device=device)
            behavior_probs.scatter_(1, actions, 1 - epsilon * (action_probs.shape[1] - 1))
            behavior = Categorical(probs=behavior_probs)

        return behavior

    def update_nu(self, kl_div: torch.Tensor, is_clip: bool):
        nu = torch.clamp(self.log_nu.exp(), min=0.0, max=1000000.0)
        nu_loss = -nu * (kl_div.detach() - self.bc_kl_beta)
        self.nu_optimizer.zero_grad()
        nu_loss.backward()
        if is_clip:
            self.log_nu.grad.data.clamp_(-1, 1)
        self.nu_optimizer.step()
        return {
            'nu_loss': nu_loss.detach().cpu().item(),
            'nu': nu.detach().cpu().item(),
        }


class BCE(BC):
    def __init__(
        self, 
        device: torch.device, 
        bc_type: str, 
        bc_kl_beta: float, 
        use_pi_b_kl: bool, 
        nu_lr: float,
        sofa_threshold: int,
        use_sofa_cv: bool,
        is_sofa_threshold_below: bool,
        kl_threshold_type: str,
        kl_threshold_exp: float,
        kl_threshold_coef: float,
        pi_b_model_path: str = 'pi_b_models/model_mean_agg.pth'
    ):
        super().__init__(device, bc_type, bc_kl_beta, use_pi_b_kl, nu_lr, pi_b_model_path)
        self.sofa_threshold = sofa_threshold
        self.use_sofa_cv = use_sofa_cv
        self.is_sofa_threshold_below = is_sofa_threshold_below
        self.kl_threshold_type = kl_threshold_type
        self.kl_threshold_exp = kl_threshold_exp
        self.kl_threshold_coef = kl_threshold_coef


    def compute_kl_threshold(
        self, 
        shape: torch.Size, 
        bc_condition: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        if self.kl_threshold_type == 'step':
            # add 1 to avoid threshold be 0
            kl_threshold = torch.full(shape, self.bc_kl_beta, device=device) \
                            * (bc_condition.view(-1,) + 1)
        elif self.kl_threshold_type == 'exp':
            kl_threshold = self.kl_threshold_coef \
                            * (torch.full(shape, self.kl_threshold_exp, device=device) ** bc_condition.view(-1,))
        else:
            raise ValueError("Wrong kl threshold type!")
        return kl_threshold.detach()


    def get_mask(self, bc_condition: torch.Tensor) -> torch.Tensor:
        '''
        Return:
            get behavior cloning condition mask
        '''
        if self.is_sofa_threshold_below:
            mask = bc_condition < self.sofa_threshold
        else:
            mask = bc_condition >= self.sofa_threshold
        return mask.to(torch.float).view(-1).detach()

    def compute_bc_loss(
        self, 
        kl_div: torch.Tensor, 
        kl_threshold: torch.Tensor, 
        bc_condition: torch.Tensor
    ) -> torch.Tensor:
        # \nu * (KL(\pi_\phi(a|s) || \pi_{clin}(a|s)) - \beta)
        nu = torch.clamp(self.log_nu.exp(), min=0.0, max=1000000.0)
        mask = self.get_mask(bc_condition)
        bc_loss = nu * ((kl_div - kl_threshold) * mask).mean()

        return bc_loss

    def update_nu(
        self, 
        kl_div: torch.Tensor, 
        kl_threshold: torch.Tensor, 
        bc_condition: torch.Tensor,
        is_clip: bool
    ):
        nu = torch.clamp(self.log_nu.exp(), min=0.0, max=1000000.0)
        mask = self.get_mask(bc_condition)
        nu_loss = -nu * ((kl_div - kl_threshold).detach() * mask).mean()
        self.nu_optimizer.zero_grad()
        nu_loss.backward()
        if is_clip:
            self.log_nu.grad.data.clamp_(-1, 1)
        self.nu_optimizer.step()
        return {
            'nu_loss': nu_loss.detach().cpu().item(),
            'nu': nu.detach().cpu().item()
        }
