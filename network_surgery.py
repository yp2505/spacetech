import torch
import zipfile
import os
import shutil
import tempfile

def pad_tensor(old_tensor, new_shape):
    new_tensor = torch.zeros(new_shape, dtype=old_tensor.dtype, device=old_tensor.device)
    if len(old_tensor.shape) == 1:
        new_tensor[:old_tensor.shape[0]] = old_tensor
    elif len(old_tensor.shape) == 2:
        new_tensor[:old_tensor.shape[0], :old_tensor.shape[1]] = old_tensor
    else:
        raise ValueError(f"Unsupported tensor dimension: {len(old_tensor.shape)}")
    return new_tensor

def perform_surgery(zip_path, output_path):
    print(f"Starting Network Surgery on {zip_path}...")
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
        
    policy_path = os.path.join(temp_dir, "policy.pth")
    old_state_dict = torch.load(policy_path, map_location="cpu", weights_only=True)
    new_state_dict = {}
    
    for name, old_tensor in old_state_dict.items():
        new_shape = list(old_tensor.shape)
        
        # ACTOR NETWORKS (was [128, 128], now [256, 256])
        if "mlp_extractor.actor_net.0.weight" in name:
            new_shape[0] = 256
        elif "mlp_extractor.actor_net.0.bias" in name:
            new_shape[0] = 256
        elif "mlp_extractor.actor_net.2.weight" in name:
            new_shape[0] = 256
            new_shape[1] = 256
        elif "mlp_extractor.actor_net.2.bias" in name:
            new_shape[0] = 256
        elif "action_net.weight" in name:
            new_shape[1] = 256
            
        # CRITIC NETWORKS (was [256, 256, 128], now [512, 512, 256])
        elif "mlp_extractor.critic_net.0.weight" in name:
            new_shape[0] = 512
        elif "mlp_extractor.critic_net.0.bias" in name:
            new_shape[0] = 512
        elif "mlp_extractor.critic_net.2.weight" in name:
            new_shape[0] = 512
            new_shape[1] = 512
        elif "mlp_extractor.critic_net.2.bias" in name:
            new_shape[0] = 512
        elif "mlp_extractor.critic_net.4.weight" in name:
            new_shape[0] = 256
            new_shape[1] = 512
        elif "mlp_extractor.critic_net.4.bias" in name:
            new_shape[0] = 256
        elif "value_net.weight" in name:
            new_shape[1] = 256
            
        if list(old_tensor.shape) != new_shape:
            print(f"Padding {name} from {list(old_tensor.shape)} to {new_shape}")
            new_state_dict[name] = pad_tensor(old_tensor, tuple(new_shape))
        else:
            new_state_dict[name] = old_tensor.clone()
            
    torch.save(new_state_dict, policy_path)
    
    print(f"Repackaging into {output_path}...")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)
                
    shutil.rmtree(temp_dir)
    print("Surgery complete!")

if __name__ == "__main__":
    if os.path.exists("ppo_satellite_1.zip"):
        perform_surgery("ppo_satellite_1.zip", "ppo_satellite_1_upgraded.zip")
        shutil.move("ppo_satellite_1_upgraded.zip", "ppo_satellite_1.zip")
    if os.path.exists("ppo_satellite_2.zip"):
        perform_surgery("ppo_satellite_2.zip", "ppo_satellite_2_upgraded.zip")
        shutil.move("ppo_satellite_2_upgraded.zip", "ppo_satellite_2.zip")
    print("Files overwritten.")
