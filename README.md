# Learn Reinforcement Learning

I will be developing this tutorial for everyone who is interested to tryout reinforcement learning through doing hands-on projects. We will start simple and eventually will build up the complexity of the project. I plan to cover some topics in **OpenAI Gym** and simulation in **Gazebo** and **Isaac Sim** using **ROS2**. Please forgive any mistakes that you find. The project structure is a bit messy since I'm also in the process of figuring it out. You're more than welcome to put issues and pull requests for improving this repository. Let's grow together!

# Sections
1. [Initial Setup](https://github.com/ashvin-a/Learn-RL#setup)
2. [Building Cartpole Agent](https://github.com/ashvin-a/Learn-RL/#1-cartpole-agent)
3. [Building Walker2D](https://github.com/ashvin-a/Learn-RL/#2-walker2d)

   
# Setup
Follow these commands for cloning and trying out this repository.
1. Let's first clone the repository using this command:
```
git clone git@github.com:ashvin-a/Learn-RL.git
```
2. Create a virtual environment for installing the dependencies.
```
python3 -m venv env 
```
3. Utilise the environment. For a Windows machine, run:
```
.\env\Scripts\activate
```
And for Linux/Mac, run:
```
source env/bin/activate
```
4. Now, let's install the dependencies.
```
pip install -r requirements.txt
```
Yay! Now you've completed the setup! You can try running the trained Cartpole agent by running:
```
python src/simulation/simulation/cartpole/cartpole_test.py
```


# Projects
## 1. Cartpole Agent 
We will be using a simple agent, i.e, `CartPole-V1`, for this project. Here, the goal will be to balance the stick like an inverted pendulum. You could check out `src/simulation/simulation/cartpole/cartpole.py` for training the agent. You could try out the model by running `cartpole_test.py`.

<img src="https://github.com/ashvin-a/Learn-RL/blob/main/src/assets/cartpole/cartpole.gif">

## 2. Walker2D
This is a little more complicated. We now have multiple joints, and we have a lot more states and actions for this agent. Let's first train the agent to stay alive and keep hoping.


