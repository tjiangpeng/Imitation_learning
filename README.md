# Future Trajectory Prediction with Deep Imitation Learning

TODO:

- [x] Generate dataset with CARLA simulator

- [x] One rendered image --> CNN --> Fully connected layers --> Future Trajectory

  - [x]  VGG-16 architecture
  - [x] ResNet-50 architecture

- [ ] Separate the input information

  - HD Map (M)

  - Surrounding vehicles' state (S)

  - Ego state (E)

  - Routing (R)

- [ ] Input images --> CNN --> Parametric probability distribution (Mixed Gaussian Model) $\sigma$

- <img src="https://latex.codecogs.com/gif.latex?P(Y|M,S,E,R) = \sum{\phi_i \mathcal{N}(\mu_i, \sigma_i^2)}" />
  
  Loss is to minimize the negative log-likelihood of the ground truth future locations under the predicted trajectory according to a GMM with parameters 

  don't know many latent factors should be defined?

  - [ ] Remove Routing information

- [ ] Input continuous frames (Video) 
