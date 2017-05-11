# RattLe
*Slither.io reinforcement learning agent*

## Setup instructions: ##
1.) Create a Python 2.7 (virt env)
'''
conda create python=2.7 --name rattle
'''

2.) install system stuff

```
brew install golang libjpeg-turbo
```

3.) install [Docker](https://docs.docker.com/engine/installation/)

4.) init env
```
source activate rattle
pip install -r requirements.txt
```
5.) get universe
```
git clone https://github.com/openai/universe.git
cd universe
pip install -e .
```
6.) test system
```
cd ..
python test.py
```
