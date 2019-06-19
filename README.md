# Differential-Programming

关于Differential Programming/Differentiable Programming/DDP/Automatic differentiation/Differentiable neural computer的论文，文章，教程，幻灯片和项目的列表。<br>
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

“Differentiable programming”: This is the idea of viewing a program (or a circuit) as a graph of differentiable modules that can be trained with backprop. This points towards the possibility of not just learning to recognize patterns (as with feed-forward neural nets) but to produce algorithms (with loops, recursion, subroutines, etc). There are a few papers on this from DeepMind, FAIR, and others, but it’s rather preliminary at the moment.
“差分编程”：这是将程序（或电路）视为可以用backprop训练的可微模块的图形的想法。 这表明不仅可以学习识别模式（如前馈神经网络），还可以生成算法（包括循环，递归，子程序等）。 有较少来自于DeepMind，FAIR和其他的文章，但目前处于初步阶段。

* [Age of AI Talk:Deep Learning est Mort! Vive Differentiable Programming](https://medium.com/amplify-partners/age-of-ai-talk-deep-learning-est-morte-vive-differentiable-programming-6b1a1c9800d8)<br>

* [Neural Networks, Types, and Functional Programming](http://colah.github.io/posts/2015-09-NN-Types-FP/)<br>
* [Differentiable neural computer](https://en.wikipedia.org/wiki/Differentiable_neural_computer)<br>
* [Automatic differentiation](https://en.m.wikipedia.org/wiki/Automatic_differentiation)<br>
* [Introduction to AUTOMATIC DIFFERENTIATION](https://alexey.radul.name/ideas/2013/introduction-to-automatic-differentiation/)


Papers
-------
* Differential Programming

  [1][Atkeson C G, Stephens B J. Random sampling of states in dynamic programming[J]. IEEE Trans Syst Man Cybern B Cybern, 2008, 38(4):924-929.](https://ieeexplore.ieee.org/document/4559368/citations)<br>
  [2][Fan D D, Theodorou E A. Differential Dynamic Programming for time-delayed systems[C]// Decision and Control. IEEE, 2016:573-579.](https://doi.org/10.1109/CDC.2016.7798330)<br>
  [3][Pan Y, Theodorou E A. Probabilistic differential dynamic programming[J]. Advances in Neural Information Processing Systems, 2014, 3:1907-1915.](https://papers.nips.cc/paper/5248-probabilistic-differential-dynamic-programming)<br>
  [4][Yamaguchi A, Atkeson C G. Neural networks and differential dynamic programming for reinforcement learning problems[C]// IEEE International Conference on Robotics and Automation. IEEE, 2016.](https://doi.org/10.1109/ICRA.2016.7487755)<br>
  [5][Mayne D Q. Differential Dynamic Programming - A Unified Approach to the Optimization of Dynamic Systems[J]. Control & Dynamic Systems, 1973, 10:179-254.](https://doi.org/10.1016/B978-0-12-012710-8.50010-8)<br>
  [6][DehazeGAN: When Image Dehazing Meets Differential Programming,IJCAI-18 ](https://doi.org/10.24963/ijcai.2018/172)<br>
更新于2019.6.19<br>
  [7][Zhang_ISTA-Net_Interpretable_Optimization-Inspired_CVPR_2018_paper]<br>
  [8][ISTA-Net Iterative Shrinkage-Thresholding Algorithm Inspired Deep Network for Image Compressive Sensing]<br>
  [9][Deep ADMM-Net for Compressive Sensing MRI]<br>
  [10][Deep Unfolding Model-Based Inspiration of Novel Deep Architectures]<br>
  [11][Differentiable Linearized ADMM]<br>
  [12][Differentiable Optimization-Based Modeling for Machine Learning ](https://github.com/bamos/thesis)<br>
* Differentiable Programming
 
  [1][Singh C. Optimality conditions in multiobjective differentiable programming[J]. Journal of Optimization Theory & Applications, 1988, 57(2):369-369.](https://doi.org/10.1007/BF00938820)<br>
  [2][Wang F, Wu X, Essertel G, et al. Demystifying Differentiable Programming: Shift/Reset the Penultimate Backpropagator[J]. 2018.](https://arxiv.org/abs/1803.10228)<br>
  [3][Tzu-Mao Li, Michaël Gharbi, Andrew Adams, Frédo Durand, Jonathan Ragan-Kelley. Differentiable Programming for Image Processing and Deep Learning in Halide. ACM Transactions on Graphics 37(4) (Proceedings of ACM SIGGRAPH 2018) ](https://people.csail.mit.edu/tzumao/gradient_halide/)[code](https://github.com/jrk/gradient-halide)<br>
  [4][A Language and Compiler View on Differentiable Programming ICLR 2018 Workshop Track ](https://openreview.net/forum?id=SJxJtYkPG)<br>
  [5][Gaunt A L, Brockschmidt M, Kushman N, et al. Differentiable Programs with Neural Libraries[J]. PMLR.2017.](http://proceedings.mlr.press/v70/gaunt17a.html)<br>
  [6][k-meansNet: When k-means Meets Differentiable Programming arxiv 2018](https://arxiv.org/pdf/1808.07292v1.pdf)<br>
  [7][Differentiable Learning of Logical Rules for Knowledge Base Reasoning NIPS 2017](https://papers.nips.cc/paper/6826-differentiable-learning-of-logical-rules-for-knowledge-base-reasoning.pdf)<br>
  [8][Programming With a Differentiable Forth Interpreter](https://openreview.net/pdf?id=HkJq1Ocxl)<br>
  [9][Efficient Differentiable Programming in a Functional Array-Processing Language](https://arxiv.org/abs/1806.02136)<br>
  
 * Automatic differentiation
 [1][Baydin A G, Pearlmutter B A, Radul A A, et al. Automatic differentiation in machine learning: a survey[J]. Computer Science, 2015(February).](https://arxiv.org/abs/1502.05767)<br>

  
slides
-------
* Differential Programming

  [1][Atkeson_Stephens_Random Sampling of States in Dynamic Programming_NIPS 2007](https://pdfs.semanticscholar.org/presentation/9b05/8eb3539b894d1433113f7f6fee8b8e337a7e.pdf)<br>
  [2][Stochastic Differential Dynamic Programming_Theodorou_ACC2010](https://homes.cs.washington.edu/~todorov/papers/TheodorouACC10.pdf)<br>
  
* Differentiable Programming

  [1][Differentiable Programming](https://www.cs.nuim.ie/~gunes/files/Baydin-MSR-Slides-20160201.pdf)<br>
  [2][Tzu-Mao Li, Michaël Gharbi, Andrew Adams, Frédo Durand, Jonathan Ragan-Kelley. Differentiable Programming for Image Processing and Deep Learning in Halide. ACM Transactions on Graphics 37(4) (Proceedings of ACM SIGGRAPH 2018) ](https://people.csail.mit.edu/tzumao/gradient_halide/)<br>
  [3][Deep learning frameworks and differentiable programming](Deep learning frameworks and differentiable programming)<br>
  [4][Differentiable Functional Programming](http://www.robots.ox.ac.uk/~gunes/assets/pdf/baydin-2016-slides-functionallondoners.pdf)<br>
  [5][Getting started with Differentiable Programing](http://nramm.nysbc.org/wp-content/uploads/2018/04/nramm_tegunov.pdf)<br>
  [6][Deep Learning, differentiable programming, and software 2.0](http://dic.uqam.ca/upload/files/seminaires/Deep%20Learning%20and%20Differentiable%20programming.pdf)<br>
  
  
material
-------
* Differential Programming

  [1][Programming of ordinary differential equations](http://hplgit.github.io/primer.html/doc/pub/ode2/ode2-readable.html)<br>
  
* Differentiable Programming

  [1][Differentiable Programming: A Semantics Perspective](https://barghouthi.github.io/2018/05/01/differentiable-programming/)<br>
  [2][Tensorlang, a differentiable programming language based on TensorFlow](https://github.com/tensorlang/tensorlang)<br>
  [3][Differentiable programming in Gluon and Python (For not only for medical image analysis)](https://github.com/jmargeta/PyConSK2018)<br>
  [4][torchbearer](https://github.com/ecs-vlc/torchbearer)<br>
  [5][Computers May Be Closer to Learning Common Sense Than We Think](https://www.huffingtonpost.com/quora/computers-may-be-closer-t_b_11318132.html)<br>
