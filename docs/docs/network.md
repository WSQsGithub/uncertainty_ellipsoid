# 2-Stage Network Design

By utilizing the 2-stage training pipeline, we can transform a multi-object optimization problem into a series of single-object optimization problems. This saves us from the pain of tuning the weights of the loss function.

## Stage One: Learning the Ellipsoid Center

Loss Function:
$$
L_1 = MSR(\hat{c}, \bar{c}) + \lambda_{reg} L_{reg}
$$
