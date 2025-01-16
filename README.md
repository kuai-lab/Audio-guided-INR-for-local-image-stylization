# Audio-guided implicit neural representation for local image stylization (Computational Visual Media 2024)
PDF: [link](https://link.springer.com/article/10.1007/s41095-024-0413-5)

## Getting Started
### Installation
- Follow the steps below:
```bash
$ conda create -n audioINR python=3.8
$ conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

### Download Pretrained Model
Download Link : [AudioEncoder](https://drive.google.com/file/d/1e_-2du7KUvadAl0_BGf5VUwgpSEe0ntL/view?usp=sharing)

Place downloaded weights under "./weights" folder.

### Download Dataset
1. [VGG-Sound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)
2. [the Greatest Hits](https://andrewowens.com/vis/)


## Stylization
### Localizing with text condition
#### Stylizing with audio
```bash
$ python train_text_audio.py --content_path "./test_set/chicago.jpg" --content_name "buildings" --audio_path "./audiosample/fire.wav"
```
#### Stylizing with text
```bash
$ python train_text_text.py --content_path "./test_set/church.jpeg" --content_name "church" --text "wood"
```

## Citations

```bibtex
@article{lee2024audio,
  title={Audio-guided implicit neural representation for local image stylization},
  author={Lee, Seung Hyun and Kim, Sieun and Byeon, Wonmin and Oh, Gyeongrok and In, Sumin and Park, Hyeongcheol and Yoon, Sang Ho and Hong, Sung-Hee and Kim, Jinkyu and Kim, Sangpil},
  journal={Computational Visual Media},
  volume={10},
  number={6},
  pages={1185--1204},
  year={2024},
  publisher={Springer}
}
```
