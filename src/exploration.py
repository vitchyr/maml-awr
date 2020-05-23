
def policy_advantage_on_batch(self, policy: nn.Module, value_function: nn.Module, batch: torch.tensor):
    with torch.no_grad():
        value_estimates = value_function(batch[:,:self._observation_dim])
        mc_value_estimates = self.mc_value_estimates_on_batch(value_function, batch)

        exp_clipped_advantages = (mc_value_estimates - value_estimates).squeeze(-1).clamp(max=self._advantage_clamp).exp()

        original_obs = batch[:,:self._observation_dim]
        original_action = batch[:,self._observation_dim:self._observation_dim + self._action_dim]
        original_log_probs = batch[:,-5]

        action = policy(original_obs)
        log_probs = D.Normal(action, torch.empty_like(action).fill_(self._action_sigma)).log_prob(original_action).sum(-1)
        ratio = (log_probs - original_log_probs).exp()
        clipped_ratio = ratio.clamp(min=1-self._args.ratio_clip, max=1+self._args.ratio_clip)

    if clipped_ratio.dim() > 1 or exp_clipped_advantages.dim() > 1:
        raise RuntimeError()

    return (clipped_ratio * exp_clipped_advantages).mean().item()

#################################################################
#################################################################

def train_awr_exploration(self, train_step_idx: int, batches: torch.tensor, meta_batches: torch.tensor,
                          adaptation_policies: list, value_functions: list, writer: SummaryWriter):
    exploration_loss = 0
    for i, (meta_batch, pyt_batch) in enumerate(zip(meta_batches, batches)):
        weights = []
        for policy, vf in zip(adaptation_policies[i], value_functions[i]):
            weight = self.policy_advantage_on_batch(policy, vf, meta_batch)
            weights.append(weight)

        print_(i, train_step_idx, weights, self._silent)

        for j, batch in enumerate(pyt_batch):
            batch = batch.view(-1, batch.shape[-1])
            original_action = batch[:,self._observation_dim:self._observation_dim + self._action_dim]
            original_obs = batch[:,:self._observation_dim]
            action = self._exploration_policy(original_obs)
            log_probs = D.Normal(action, torch.empty_like(action).fill_(self._action_sigma)).log_prob(original_action).sum(-1)

            loss = -log_probs.mean() * weights[j]
            exploration_loss = exploration_loss + loss

    exploration_loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(self._exploration_policy.parameters(), self._grad_clip)
    writer.add_scalar(f'Explore_Grad', grad, train_step_idx)
    writer.add_scalar(f'Explore_Loss', exploration_loss.item(), train_step_idx)

    self._exploration_policy_optimizer.step()
    self._exploration_policy_optimizer.zero_grad()


#@profile
def train_vae_exploration(self, train_step_idx: int, writer: SummaryWriter):
    grads = []
    losses = []
    recon_losses = []
    kld_losses = []
    mu_ys = []
    std_ys = []
    mu_zs = []
    std_zs = []
    for step in range(self._args.vae_steps):
        kld_loss = 0
        recon_loss = 0
        for i, buf in enumerate(self._outer_buffers):
            batch, trajectories = buf.sample(self._args.exploration_batch_size, trajectory=True, complete=True)
            batch = torch.tensor(batch, device=self._device)

            trajectories = torch.tensor(trajectories, device=self._device)
            traj = trajectories[:,:,:self._observation_dim + self._action_dim].permute(0,2,1)[:,:,::self._args.cvae_skip]

            obs = batch[:,:self._observation_dim]
            action = batch[:,self._observation_dim:self._observation_dim + self._action_dim]

            pz_t, z = self._exploration_policy.encode(traj, sample=True)
            pz = self._exploration_policy.prior(obs)

            py_zs = self._exploration_policy.decode(z, obs)
            mu_y = py_zs[:,:py_zs.shape[-1]//2]
            std_y = (py_zs[:,py_zs.shape[-1]//2:] / 2).exp()

            d_y = D.Normal(mu_y, std_y)

            kld_loss += kld(pz_t, pz).mean()
            recon_loss += -d_y.log_prob(action).sum(-1).mean()

        #kld_coef = (torch.tensor(train_step_idx).float() / 10000 - 10).sigmoid().to(self._device)
        kld_coef = self._args.kld_coef

        exploration_loss = kld_coef * kld_loss + recon_loss
        exploration_loss.backward()

        grad = torch.nn.utils.clip_grad_norm_(self._exploration_policy.parameters(), self._grad_clip)
        self._exploration_policy_optimizer.step()
        self._exploration_policy_optimizer.zero_grad()

        grads.append(grad)
        losses.append(exploration_loss.item())
        recon_losses.append(recon_loss.item())
        kld_losses.append(kld_loss.item())
        mu_ys.append(mu_y.detach().cpu().numpy())
        std_ys.append(std_y.detach().cpu().numpy())
        mu_zs.append(pz_t[:,:pz_t.shape[-1]//2:].detach().cpu().numpy())
        std_zs.append(pz_t[:,pz_t.shape[-1]//2:].div(2).exp().detach().cpu().numpy())

    writer.add_scalar(f'Explore_Grad', np.mean(grads), train_step_idx)
    writer.add_scalar(f'Explore_Loss', np.mean(losses), train_step_idx)
    writer.add_scalar(f'Explore_Loss_Recon', np.mean(recon_losses), train_step_idx)
    writer.add_scalar(f'Explore_Loss_KLD', np.mean(kld_losses), train_step_idx)
    writer.add_histogram(f'Explore_Mu', np.mean(mu_ys), train_step_idx)
    writer.add_histogram(f'Explore_Sigma', np.mean(std_ys), train_step_idx)
    writer.add_histogram(f'Explore_ZMu', np.mean(mu_zs), train_step_idx)
    writer.add_histogram(f'Explore_ZSigma', np.mean(std_zs), train_step_idx)
