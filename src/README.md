## Proximal Policy Optimization with Generalized Advantage Estimation

By Patrick Coady: [Learning Artificial Intelligence](https://learningai.io/)

(hardmaru cloned the original repo and added support for roboschool, and saving model weights so they can be played back later using `python demo.py path_to_json`.)

Link to original [repo](https://github.com/pat-coady/trpo).

I have been able to use Pat's PPO implementation to train on OpenAI's [roboschool](https://blog.openai.com/roboschool/) environments (which are open source, and don't require mujoco). They are also tougher to train compared to the original environments. I have also made it work for pybullet's [environments](pybullet.org) as well (such as the racecar and minitaur).

A few changes:

- rather than using a factor of 10 to decide size of net, it is a hyperparameter for train.py, and I found using a factor of 5 to be sufficient for most roboschool tasks.
- I made the log_std bias term a hyperparameter (it was originally set to -1.0). I noticed that making it -0.5 or even 0.0 helps with the exploration for tougher tasks. If it is set to -1.0, roboschool's harder version of walker2d and hopper won't train. It gets stuck at standing around the starting position.
- `python demo.py zoo/humanoid.json` is used to playback trained models (dumped as .json weights). I was able to use PPO to train the roboschool humanoid, ant, reacher, bipedhard, etc, and put the final set of weights in the `zoo` directory to play back afterwards. `demo.py` is independent of TensorFlow, and is a pure `numpy` implementation of the feed forward network policy.
