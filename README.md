# Learn Reinforcement Learning

I will be developing this tutorial for everyone who is interested to tryout reinforcement learning through doing hands-on projects. We will start simple and eventually will build up the complexity of the project. I plan to cover some topics in OpenAI Gym and simulation in Gazebo and Isaac Sim using ROS2. Please forgive any mistakes that you find. The project structure is a bit messy since I'm also in the process of figuring it out. You're more than welcome to put issues and pull requests for improving this repository. Let's grow together!

## 1. Cartpole Agent 
We will be using a simple agent, i.e, `CartPole-V1` for this project. Here, the goal will be to balance the stick like an inverted pendulum. You could check out `src/simulation/simulation.cartpole.py` for training the agent. You could tryout the model by running `cartpole_test.py`

## 2. Walker2D
This is a little more complicated. We now have multiple joints and we have lot more states and actions for this agent. Let's first write a script for it to stand still.


