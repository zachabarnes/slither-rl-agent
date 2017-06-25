# RattLe: a Slither.io reinforcement learning agent
##### By [Zach Barnes](https://github.com/zabarnes), [Tyler Romero](https://github.com/tyler-romero), [Frank Cipollone](https://github.com/fcipollone).

### Installation Instructions (meant for ubuntu VM):
- Install [Conda](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04) for ubuntu 16.04

- Create Conda env
```
conda create --name rattle python=3.5
```

- Activate a conda env
```
source activate rattle
```

- Install needed packages
```
sudo apt-get update
sudo apt-get install -y tmux htop cmake golang libjpeg-dev libgtk2.0-0 ffmpeg
```

- Install universe installation dependencies
```
pip install numpy
```

- Install universe
```
git clone https://github.com/openai/universe.git
cd universe
pip install -e .
```
- Install codebase and packages
```
cd ..
git clone https://github.com/zabarnes/RattLe.git
cd RattLe
pip install -r requirements.txt
```

- Install [docker](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04) for ubuntu 16.04 **MAKE SURE TO DO STEP 2 AS WELL**

- Restart VM

### Test installation

Run the test agent script
```
cd RattLe
python test.py
```
you should see a tiny rendering of the game or "yay" on the command line.

### Train a model

Run the corresponding shell script. For example, to train our Recurrent Q model, run:
```
train_recurrentq.sh
```
