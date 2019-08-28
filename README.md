# Future Trajectory Prediction with Deep Imitation Learning

TODO:

- [x] Generate dataset with CARLA simulator

- [x] One rendered image --> CNN --> Fully connected layers --> Future Trajectory

  - [x]  VGG-16 architecture
  - [x] ResNet-50 architecture

- [ ] Separate the input information which can avoid overlapping problem

  - HD Map (M)
- Surrounding vehicles' state (S)
  - Ego state (E)
- Routing (R)
  
- [ ] Trained with continuous frames (every one second) 
- [ ] Input images --> CNN --> Parametric probability distribution (Mixed Gaussian Model)

![equation](https://latex.codecogs.com/svg.latex?P(Y|M,S,E,R)&space;=&space;\sum{\phi_i&space;\mathcal{N}(\mu_i,&space;\sigma_i^2)})

  Loss is to minimize the negative log-likelihood of the ground truth future locations under the predicted trajectory according to a GMM with parameters 

  - [ ] Try to output probabilistic grid map

    explicit theory need to be stated ...

## Rendered Image

Using 



## VGG-based Network

