import argparse
import torch

import torch.distributions as D
import torch.optim as O

from src.nn import CVAE
from src.utils import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--buffer_paths', type=str, nargs='+', default=None)
    parser.add_argument('--state_dim', type=int)
    parser.add_argument('--action_dim', type=int)
    parser.add_argument('--info_dim', type=int)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--trajectory_skip', type=int, default=10)
    parser.add_argument('--kld_coef', type=float, default=1.)
    parser.add_argument('--log_path', type=str)
    return parser.parse_args()


def train_vae_exploration(self, policy: nn.Module, optimizer: torch.optim.Optimizer,
                          writer: SummaryWriter, args: argparse.Namespace, buf: ReplayBuffer):
    policy.to(args.device)
    losses = []
    recon_losses = []
    kld_losses = []
    mu_ys = []
    std_ys = []
    mu_zs = []
    std_zs = []
    for step in range(args.steps):
        kld_loss = 0
        recon_loss = 0

        batch, trajectories = buf.sample(args.batch_size, trajectory=True, complete=True)
        batch = torch.tensor(batch, device=args.device)

        trajectories = torch.tensor(trajectories, device=args.device)
        traj = trajectories[:,:,:args.state_dim + args.action_dim].permute(0,2,1)[:,:,::args.trajectory_skip]

        obs = batch[:,:args.state_dim]
        action = batch[:,args.state_dim:args.state_dim + args.action_dim]

        pz_t, z = policy.encode(traj, sample=True)
        pz = policy.prior(obs)

        py_zs = policy.decode(z, obs)
        mu_y = py_zs[:,:py_zs.shape[-1]//2]
        std_y = (py_zs[:,py_zs.shape[-1]//2:] / 2).exp()

        d_y = D.Normal(mu_y, std_y)

        kld_loss += kld(pz_t, pz).mean()
        recon_loss += -d_y.log_prob(action).sum(-1).mean()

        exploration_loss = args.kld_coef * kld_loss + recon_loss
        exploration_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses.append(exploration_loss.item())
        recon_losses.append(recon_loss.item())
        kld_losses.append(kld_loss.item())
        mu_ys.append(mu_y.detach().cpu().numpy())
        std_ys.append(std_y.detach().cpu().numpy())
        mu_zs.append(pz_t[:,:pz_t.shape[-1]//2:].detach().cpu().numpy())
        std_zs.append(pz_t[:,pz_t.shape[-1]//2:].div(2).exp().detach().cpu().numpy())

    writer.add_scalar(f'Explore_Loss', np.mean(losses), train_step_idx)
    writer.add_scalar(f'Explore_Loss_Recon', np.mean(recon_losses), train_step_idx)
    writer.add_scalar(f'Explore_Loss_KLD', np.mean(kld_losses), train_step_idx)
    writer.add_histogram(f'Explore_Mu', np.mean(mu_ys), train_step_idx)
    writer.add_histogram(f'Explore_Sigma', np.mean(std_ys), train_step_idx)
    writer.add_histogram(f'Explore_ZMu', np.mean(mu_zs), train_step_idx)
    writer.add_histogram(f'Explore_ZSigma', np.mean(std_zs), train_step_idx)


def combine_buffers(args: argparse.Namespace) -> ReplayBuffer:
    trajectories = np.concatenate([np.load(path) for path in args.buffer_paths])
    new_buffer = ReplayBuffer(trajectories.shape[1], args.state_dim, args.action_dim, args.info_dim, trajectories.shape[0],
                              args.discount_factor, True)
    new_buffer._trajectories = trajectories
    new_buffer._stored_trajectories = trajectories.shape[0]
    
    return new_buffer


def run(args: argparse.Namespace):
    policy = CVAE(args.state_dim, args.action_dim, args.latent_dim).to(args.device)
    optimizer = O.Adam(policy.parameters(), lr=args.lr)
    buf = combine_buffers(args)
    writer = SummaryWriter(args.log_path)
    train_vae_exploration(policy, optimizer, writer, args, buf)

if __name__ == '__main__':
    run(get_args())
