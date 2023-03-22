# [NTIRE 2023 Challenge on Image Super-Resolution (x4)](https://cvlai.net/ntire/2023/) @ [CVPR 2023](https://cvpr2023.thecvf.com/)

## Latent Discriminative Cosine Criterion (LDCC) Overview
<img src="https://github.com/kimtaehyeong/NTIRE2023_ImageSR_x4_LDCC/blob/main/figures/ldcc_method.PNG" width="600"/>

### Environment
- [PyTorch >= 1.7](https://pytorch.org/) **(Recommend **NOT** using torch 1.8!!! It would cause abnormal performance.)**
- [BasicSR == 1.3.4.9](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 

### How to test the baseline model?

1. `git clone https://github.com/kimtaehyeong/NTIRE2023_ImageSR_x4.git`
2. The models weights should download to ```model_zoo``` directory at [Google Drive](https://drive.google.com/file/d/1UqM1tU09TOFO-E_Y5lQpm5pjXmAdTGC-/view?usp=share_link)
3. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id [number]
    ```
    for example :
    ```bash
     CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir /home/work/NTIRE/dataset/SUB --save_dir ./results --model_id 2
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.
    - This repository is based on a [link](https://github.com/zhengchen1999/NTIRE2023_ImageSR_x4).
    - We provide a baseline of our model based on [HAT](https://github.com/XPixelGroup/HAT). The code and pretrained models of our models are provided. Our baseline are all test normally with `run.sh`.

  
### How to calculate the number of parameters, FLOPs, and activations

```python
    from utils.model_summary import get_model_flops, get_model_activation
    from models.team00_RFDN import RFDN
    model = RFDN()
    
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
```
### Acknowledgement
This code is built on [HAT](https://github.com/XPixelGroup/HAT) codebase. We thank the authors for sharing the codes.

### Team
[Jungkeong Kil](https://github.com/kil-jung-keong),
Eon Kim,
[Taehyung Kim](https://github.com/kimtaehyeong),
[Yeonseung Yu](https://github.com/yuyeonseung),
[Beomyeol Lee](https://github.com/by2ee),
[Subin Lee](https://github.com/Leebsun),
[Seokjae Lim](https://github.com/SeokjaeLIM),
[Somi Chae](https://github.com/csi714),
[Heungjun Choi](https://github.com/hjvision96)

### License
This code repository is release under [MIT License](LICENSE). 
