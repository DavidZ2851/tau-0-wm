
# $\tau_0$-World Model


![Overview](figures/VAM-teaser-img.jpg)


This repo is the official implementation of **$\tau_0$-World Model: A Unified Video-Action World Model forRobotic Manipulation**.


## News

- [2026.06.01] 🚀 We release [$\tau_0$-World Model](link).


## Pretrained Model

* The pretrained weights of VAM can be found on [$\tau_0$-WM](https://huggingface.co/).

* The pretrained weights of Simulator will be released soon.

* The inference codes of Test-Time Computation 


## Real-World Deployment


### Preparation

1. Download the pretrained weight of [$\tau_0$-WM](https://huggingface.co/)

2. Download the weight of [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)

3. Replace `diffusion_model.model_path` in `configs/deployment/wan_pretrain_rela_eef6d.yaml` with your local path to $\tau_0$-WM's weight

4. Replace `vae_path` in the config with your local path to VAE's weight

5. Replace `text_encoder.checkpoint_path` and `text_encoder.tokenizer_path` in the config with your local path to text encoder and tokernizer


### Action Space
In the pretraining stage, $\tau_0$-WM's is optimized to predict the relative pose of end effectors, including 20 dimensions (xyz and 6d-rotation for each arm). The coordinate origin of each eef pose is the current **Arm Base link**.

### Running
We provide a simple script of deploying $\tau_0$-WM server based on :

```
# Policy Server
bash web_infer_scripts/run_server.sh $HOST $PORT
```

```
# A simple client that send random observations
python web_infer_utils/simple_client_wan.py
```



## Acknowledgment

- The video model of $\tau_0$-WM is built on [Wan-2.2](https://github.com/Wan-Video/Wan2.2).
- The web-socket based policy server is built on [openpi](https://github.com/Physical-Intelligence/openpi).

