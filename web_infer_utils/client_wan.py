import dataclasses

from web_infer_utils.openpi_client import websocket_client_policy as _websocket_client_policy
import numpy as np
import logging
import tyro
import time
from utils import init_logging, import_custom_class, save_video
from yaml import load, dump, Loader, Dumper, safe_load
from utils.extra_utils import act_metric, save_two_tensors_by_channel
import os
import torch
from statistics import mean


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    host: str = "localhost"
    # Port to connect to the server. If None, the server will use the default port.
    port: int | None = 8001
    # Number of steps to run the policy for.
    num_steps: int = 20
    
    data_config: str = 'PATH_TO_CONFIG'
    
    save_folder: str = 'PATH_TO_SAVE_INTERMEDIATE_RESULTS'

    warmup_steps: int = 20
    
    
def main(args: Args) -> None:
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    print(f"Server metadata: {policy.get_server_metadata()}")
    
    with open(args.data_config) as f:
        config = safe_load(f)
    data_class = config['data_class']
    data_class_path = config['data_class_path']
    
    dataset_class = import_custom_class(data_class, data_class_path)
    
    data_args = config['data']
    data_args["norm_action"] = False
    _dataset = dataset_class(**data_args)
    
    data = _dataset[30]
    
    obs = data['video'][:,:,0].transpose(0,1).numpy()
    prompt = data['caption']
    state = data['state'][0].numpy()
    gt_action = data['actions']
    
    action_list = []
    for i in range(50):
        start = time.time()
        from my_scripts.random_payload import make_random_play_payload  
        payload = make_random_play_payload()
        action = policy.infer(obs=payload)['actions']
        end = time.time()
        print(f"Step {i} time cost: {end-start}")
        
        save_two_tensors_by_channel(gt_action.unsqueeze(0), torch.tensor(action).unsqueeze(0), os.path.join(args.save_folder, f"traj_{i}.png"), "GT", "Pred", ncols=2)
        
        action_list.append(torch.tensor(action))
    torch.save(torch.stack(action_list,dim=0), os.path.join(args.save_folder, 'inferred_actions.pt'))

        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
