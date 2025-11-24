# Multi-Car Racing using DQN

<img width="100%" src="https://user-images.githubusercontent.com/11874191/98051650-5339d900-1e02-11eb-8b75-7f241d8687ef.gif"></img>
This repo. contains `Multi-Car-Racing` a multiplayer variant of original [`CarRacing-v0` environment](https://gym.openai.com/envs/CarRacing-v0/).

## Requirements

Ensure you have the following installed:

- gym==0.17.3
- gym-multi-car-racing
- tensorflow-gpu==1.15.2
- numpy==1.21.6
- opencv-contrib-python==4.11.0.86
- h5py==2.10.0
- box2d-py==2.3.5
- shapely==1.7.1
- pyglet==1.5.0


## Setup

1. Clone or download this repo.
2. Install required dependencies.

## Running the Code

### Test Trained Model

To Test the trained model , run:

```bash
python MCR_test_model.py
```

### Train Model

To train the model from scratch, run:

```bash
python dqn_multicar.py
```



