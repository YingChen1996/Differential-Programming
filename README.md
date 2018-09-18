# Differential-Programming

关于Differential Programming/Differentiable Programming/DDP的论文，文章，教程，幻灯片和项目的列表。<br>
* What is differential programming? How is it related to functional programming?<br>
  Differential programming, also known as Dynamic Differential Programming (DDP) is an optimization procedure for path planning used in control theory and robotics:<br>
  DDP is an algorithm that solves locally-optimal trajectories given a cost function over some space. In essence it works by locally-approximating the cost function at each point in the trajectory. It uses this approximation to finds the optimal change to the trajectory (via a set of actions) that minimizes some cost metric (e.g. cumulative cost). In the limit it converges to the optimal trajectory. [摘自https://www.quora.com/What-is-differential-programming-How-is-it-related-to-functional-programming](https://www.quora.com/What-is-differential-programming-How-is-it-related-to-functional-programming)
  
* ['Deep Learning est mort. Vive Differentiable Programming'](https://techburst.io/deep-learning-est-mort-vive-differentiable-programming-5060d3c55074)<br>
Yann LeCun:OK, Deep Learning has outlived its usefulness as a buzz-phrase.<br>
Deep Learning est mort. Vive Differentiable Programming!<br>

Yeah, Differentiable Programming is little more than a rebranding of the modern collection Deep Learning techniques, the same way Deep Learning was a rebranding of the modern incarnations of neural nets with more than two layers.<br>

But the important point is that people are now building a new kind of software by assembling networks of parameterized functional blocks and by training them from examples using some form of gradient-based optimization.<br>

An increasingly large number of people are defining the networks procedurally in a data-dependent way (with loops and conditionals), allowing them to change dynamically as a function of the input data fed to them. It's really very much like a regular progam, except it's parameterized, automatically differentiated, and trainable/optimizable. Dynamic networks have become increasingly popular (particularly for NLP), thanks to deep learning frameworks that can handle them such as PyTorch and Chainer (note: our old deep learning framework Lush could handle a particular kind of dynamic nets called Graph Transformer Networks, back in 1994. It was needed for text recognition).<br>

People are now actively working on compilers for imperative differentiable programming languages. This is a very exciting avenue for the development of learning-based AI.<br>

Important note: this won't be sufficient to take us to "true" AI. Other concepts will be needed for that, such as what I used to call predictive learning and now decided to call Imputative Learning. More on this later<br>
Papers
-------
* Differential Programming

  [1][Atkeson C G, Stephens B J. Random sampling of states in dynamic programming[J]. IEEE Trans Syst Man Cybern B Cybern, 2008, 38(4):924-929.](https://ieeexplore.ieee.org/document/4559368/citations)<br>
  [2][Fan D D, Theodorou E A. Differential Dynamic Programming for time-delayed systems[C]// Decision and Control. IEEE, 2016:573-579.](https://doi.org/10.1109/CDC.2016.7798330)<br>
  [3][Pan Y, Theodorou E A. Probabilistic differential dynamic programming[J]. Advances in Neural Information Processing Systems, 2014, 3:1907-1915.](https://papers.nips.cc/paper/5248-probabilistic-differential-dynamic-programming)<br>
  [4][Yamaguchi A, Atkeson C G. Neural networks and differential dynamic programming for reinforcement learning problems[C]// IEEE International Conference on Robotics and Automation. IEEE, 2016.](https://doi.org/10.1109/ICRA.2016.7487755)<br>
  [5][Mayne D Q. Differential Dynamic Programming - A Unified Approach to the Optimization of Dynamic Systems[J]. Control & Dynamic Systems, 1973, 10:179-254.](https://doi.org/10.1016/B978-0-12-012710-8.50010-8)<br>
  
* Differentiable Programming
 
  [1][Singh C. Optimality conditions in multiobjective differentiable programming[J]. Journal of Optimization Theory & Applications, 1988, 57(2):369-369.](https://doi.org/10.1007/BF00938820)<br>
  
slides
-------
* Differential Programming

  [1][Atkeson_Stephens_Random Sampling of States in Dynamic Programming_NIPS 2007](https://pdfs.semanticscholar.org/presentation/9b05/8eb3539b894d1433113f7f6fee8b8e337a7e.pdf)<br>
  [2][Stochastic Differential Dynamic Programming_Theodorou_ACC2010](https://homes.cs.washington.edu/~todorov/papers/TheodorouACC10.pdf)<br>
* Differentiable Programming

  [1][Differentiable Programming](https://www.cs.nuim.ie/~gunes/files/Baydin-MSR-Slides-20160201.pdf)<br>
