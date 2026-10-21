"""Binary PPO in the frozen empirical replay environment, not LLM fine-tuning.

Fresh trajectories/logprobs each iteration; reward is terminal correctness only.
HANDOFF=0, CONTINUE=1; seven learned checkpoints and a forced cap at80.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical


class PPOController(nn.Module):
    def __init__(self, input_dim, width=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim,width),nn.GELU())
        self.actor = nn.Linear(width,2)
        self.critic = nn.Linear(width,1)

    def load_warmstart(self, warm):
        self.encoder.load_state_dict(copy.deepcopy(warm.encoder.state_dict()))
        self.actor.load_state_dict(copy.deepcopy(warm.actor.state_dict()))
        # Q_hindsight is NOT a policy-value critic. Initialize a new critic.
        nn.init.zeros_(self.critic.weight)
        nn.init.constant_(self.critic.bias,0.5)

    def forward(self,x):
        z = self.encoder(x)
        return self.actor(z),self.critic(z).squeeze(-1)


@torch.no_grad()
def sample_episodes(model,x,correctness):
    """One fresh stochastic trajectory/question under fixed pi_old parameters.

    Only visited observations forwarded. Padded slots are internal loss bookkeeping,
    NOT policy features or availability masks. No policy action at K80.
    """
    n = len(x)
    if x.ndim != 3 or x.shape[1] != 7 or correctness.shape != (n,8):
        raise ValueError('Expected complete empirical grid')
    device = x.device
    active = torch.ones(n,dtype=torch.bool,device=device)
    visited = torch.zeros((n,7),dtype=torch.bool,device=device)
    actions = torch.zeros((n,7),dtype=torch.long,device=device)
    logprobs = torch.zeros((n,7),device=device)
    old_values = torch.zeros((n,7),device=device)
    stop_indices = torch.full((n,),7,dtype=torch.long,device=device)
    for t in range(7):
        indices = torch.where(active)[0]
        if len(indices) == 0:
            break
        logits,values = model(x[indices,t])
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        visited[indices,t] = True
        actions[indices,t] = action
        logprobs[indices,t] = distribution.log_prob(action)
        old_values[indices,t] = values
        stopping = indices[action == 0]
        stop_indices[stopping] = t
        active[stopping] = False
    rewards = correctness[torch.arange(n,device=device),stop_indices].float()
    # Undiscounted MC return: all CONTINUE rewards0; only final B correctness.
    returns = rewards[:,None].expand(n,7).clone()
    advantages = (returns-old_values)*visited
    return dict(actions=actions,old_logprobs=logprobs,old_values=old_values,
        visited=visited,stop_indices=stop_indices,rewards=rewards,
        returns=returns,advantages=advantages)


def ppo_loss(model,x,episodes,tie_mask,config):
    logits,values = model(x)
    distribution = Categorical(logits=logits)
    logprobs = distribution.log_prob(episodes['actions'])
    mask = episodes['visited']
    advantages = episodes['advantages'].detach()
    ratios = (logprobs-episodes['old_logprobs'].detach()).exp()
    surrogate = torch.minimum(ratios*advantages,
        ratios.clamp(1-config['clip'],1+config['clip'])*advantages)
    # Sum decisions per episode, then average episodes: no length normalization.
    actor = -(surrogate*mask).sum()/len(x)
    value = ((values-episodes['returns'].detach()).square()*mask).sum()/mask.sum()
    entropy = (distribution.entropy()*mask).sum()/mask.sum()
    # Supervised tie auxiliary uses TRAIN counterfactual labels, not rollout rewards.
    tie = F.cross_entropy(logits[tie_mask],torch.zeros(int(tie_mask.sum()),dtype=torch.long,device=x.device)) if config['tie_weight'] and tie_mask.any() else logits.sum()*0
    total = actor+config['value_weight']*value-config['entropy_weight']*entropy+config['tie_weight']*tie
    with torch.no_grad():
        kl = (((ratios-1)-(logprobs-episodes['old_logprobs']))*mask).sum()/mask.sum()
        clipped = (((ratios-1).abs()>config['clip'])*mask).sum()/mask.sum()
    return total,dict(actor=actor,value=value,entropy=entropy,tie=tie,approx_kl=kl,clip_fraction=clipped)


@torch.no_grad()
def greedy_budgets(model,x):
    n = len(x)
    result = torch.full((n,),80,dtype=torch.long,device=x.device)
    active = torch.ones(n,dtype=torch.bool,device=x.device)
    for t in range(7):
        indices = torch.where(active)[0]
        if len(indices)==0:
            break
        logits,_ = model(x[indices,t])
        stop = indices[logits.argmax(-1)==0]
        result[stop] = (t+1)*10
        active[stop] = False
    return result


class OnlineStopper:
    """Causal runtime adapter: receives actual h_t, no outcomes/gold or future h."""
    def __init__(self,model):
        self.model = model.eval()
        self.hidden,self.decisions = [],[]

    def begin(self,hidden0):
        self.hidden = [hidden0.detach().clone()]
        self.decisions = []

    @torch.no_grad()
    def __call__(self,*,step,hidden_last_position):
        from controller.profile import raw_observation,tensor_hash
        if step != len(self.hidden) or hidden_last_position.shape[0] != 1:
            raise ValueError('Expected consecutive batch1 states after begin(h0)')
        self.hidden.append(hidden_last_position[0].detach().clone())
        if step%10:
            return False
        if step==80:
            self.decisions.append(dict(k=80,action='HANDOFF',forced=True))
            return True
        x = raw_observation(torch.stack(self.hidden),step)
        device = next(self.model.parameters()).device
        logits,_ = self.model(x[None].to(device))
        action = int(logits.argmax(-1))
        self.decisions.append(dict(k=step,action=['HANDOFF','CONTINUE_10'][action],
            forced=False,probabilities=logits.softmax(-1)[0].cpu().tolist(),
            observation_sha256=tensor_hash(x),hidden_coverage_end=step))
        return action==0
