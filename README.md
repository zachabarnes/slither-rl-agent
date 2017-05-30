# RattLe: a Slither.io reinforcement learning agent
##### By [Zach Barnes](https://github.com/zabarnes), [Tyler Romero](https://github.com/tyler-romero), [Frank Cipollone](https://github.com/fcipollone).

### Installation Instructions (meant for ubuntu VM):
- Install [Conda] for ubuntu 16.04 (https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04)

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

- Install rest of needed packages
```
pip install -r requirements.txt
```

- Install [docker](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04) for ubuntu 16.04 **Make sure to follow step 2**

- Install extra packages
```
conda install opencv
```

- Restart VM

### Test installation

Run the test agent script
```
python test.py
```
you should see "yay" on the command line
