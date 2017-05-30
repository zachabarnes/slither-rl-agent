Installation Instructions
-Install Conda 
https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04

-Create Conda env
conda create --name rattle python=3.5

-Activate conda env
source activate rattle

-Install packages
sudo apt-get update
sudo apt-get install -y tmux htop cmake golang libjpeg-dev libgtk2.0-0 ffmpeg

-Install dependencies
pip install numpy

-Install universe
git clone https://github.com/openai/universe.git
cd universe
pip install -e .

-Install rest of packages
pip install -r requirements.txt

-Install docker (make sure to follow optional step 2) (requires restart)
https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04

-Install extra packages
conda install opencv
