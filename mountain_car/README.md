# Introduction

The mountain car problem piqued my interest when learning about semi-gradient SARSA in [Sutton and Barto Chapter 10.1](http://incompleteideas.net/book/the-book-2nd.html).

I was intrigued by the need for tile-coding, and was curious if a deep neural net could overcome the need for the tile-coding. I also wanted to try out some of the other methods, like Q-learning.

I also wanted to attempt Actor-Critic, a policy gradient method which supposedly has an easier time than action-value estimation methods.


# Methods

As part of a Coursera course, I ended up manually implementing the semi-gradient SARSA algorithm, so I wanted to try using Pytorch for its autograd capabilities.

I decided to try the following algorithms:

1. Q-learning with tile coding
2. Q-learning with a deeper neural net and the raw state values
3. Policy gradient with Actor-Critic

# Results

## Q-learning with Tile coding

The following plot shows the average steps per episode for a tile encoding with 8 tilings of an 8x8 grid (i.e. 512 features). The model was a single layer. The learning rate was 0.0125 over 8 runs of 100 episodes.

![Qlearn_result](./assets/mc_q_tiled_dec01_2023_Dec_05_12_10.png)

This compares favorably to the learning rate 0.1/8 result from S&B:

![SB_compare](./assets/SB_MC_steps.png)

## Deep Q Learning


## Actor-Critic