"""Offline mock smoke test, updated only to match existing production interfaces."""
from types import SimpleNamespace
import torch
import torch.nn as nn
import numpy as np
from dt_data_processing import JSSPTrajectoryConverter, JSSPDecisionTransformerDataset, create_data_loader
from decision_transformer import create_decision_transformer_for_jssp


def make_raw_traj(T=5,state_dim=4,n_jobs=5):
    return {'states':[np.arange(state_dim)+i for i in range(T)],
            'actions':[int(i%n_jobs) for i in range(T)],
            'returns_to_go':[float(T-i) for i in range(T)],
            'timesteps':list(range(T)),'rewards':[0.0]*T}


class MockLangModel(nn.Module):
    def __init__(self,hidden_size=16):
        super().__init__()
        # Do not register self as self.model: it creates a recursive module graph.
        self.config=SimpleNamespace(hidden_size=hidden_size)
        self.linear=nn.Linear(hidden_size,hidden_size)

    def forward(self,inputs_embeds=None,**kwargs):
        return SimpleNamespace(last_hidden_state=self.linear(inputs_embeds))


def main():
    torch.manual_seed(62)
    converter=JSSPTrajectoryConverter(state_dim=4,act_dim=5)
    raw=[make_raw_traj(T=6) for _ in range(2)]
    converted=converter.convert_multiple_trajectories(raw,max_length=8)
    dataset=JSSPDecisionTransformerDataset(converted,max_length=8)
    batch=next(iter(create_data_loader(dataset,batch_size=2,shuffle=False)))
    dt=create_decision_transformer_for_jssp(
        model_name='mock',lang_model=MockLangModel(),use_peft=False,
        n_jobs=5,n_machines=5,max_length=8,max_ep_len=8,dtype='float32')
    states=torch.randn(batch['states'].shape[0],batch['states'].shape[1],dt.hidden_size)
    output=dt.forward(states,batch['returns_to_go'].float(),
        timesteps=batch['timesteps'].long(),actions=batch['actions'].long(),
        attention_mask=batch['mask'].bool())
    assert output.logits.shape==(2,8,5),output.logits.shape
    assert torch.isfinite(output.logits).all()
    loss=output.logits.square().mean()
    loss.backward()
    assert dt.action_head.weight.grad is not None
    assert torch.isfinite(dt.action_head.weight.grad).all()
    print('MOCK_FORWARD_BACKWARD_OK',tuple(output.logits.shape),float(loss))


if __name__=='__main__': main()
