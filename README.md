# uncertainty_ellipsoid

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Learning the least conservative ellipsoid region given parameter uncertainty.

## Task description

Suppose we have a studio camera with uncertain parameters and we want to estimate the uncertainty ellipsoid region that contains the true parameters with a high probability.

The uncertain parameters are the camera intrinsic parameters, which are the focal length, principal point, and distortion; and the camera extrinsic parameters, which are the rotation and translation.

We do not have the true parameters, but we have a range of possible values for each parameter.

Here is a brief description of the parameters:

```python
# 焦距范围 (f_x, f_y): 
f_x_range = (595.0, 615.0)
f_y_range = (595.0, 615.0)

# 主点范围 (c_x, c_y): 
c_x_range = (290.0, 330.0)
c_y_range = (230.0, 270.0)

# 旋转矩阵 (rx, ry, rz):
rx_range = (0.75, 1.75)
ry_range = (-1.75, -0.75)
rz_range = (0.75, 1.75)

# 平移向量 (tx, ty, tz):
tx_range = (-0.35, 0.25)
ty_range = (-0.35, 0.25)
tz_range = (-0.25, -0.05)
```

An uncertainty parameter set will be expressed as a 20-tuple: 

```python
uncertainty_param_set = (
    # Focus range (f_x, f_y)
    f_x_min, f_x_max,  
    f_y_min, f_y_max, 
    
    # Principal point range (c_x, c_y)
    c_x_min, c_x_max, 
    c_y_min, c_y_max,  
    
    # Rotation matrix (rx, ry, rz)
    rx_min, rx_max,  
    ry_min, ry_max,  
    rz_min, rz_max,  
    
    # Translation vector (tx, ty, tz)
    tx_min, tx_max, 
    ty_min, ty_max, 
    tz_min, tz_max 
)
```

where each range is a subset of possible values for the corresponding parameter. For example, `f_x_range = (600.0, 610s.0)` means that the true value of `f_x` lies within this range, and similarly for all other parameters.

Given a pixel coordinate `(u,v)` and depth `d`, we wish to estimate the possible range of world coordinates of the corresponding point, where `d` ranges from $0.2 - 0.7$ and `(u,v)` lies in a `480*640` canvas.

## Methodology

We will approximate the region with an ellipsoid, which is defines by the following equation:

$$
(x - c)^T P (x - c) \leq 1
$$

where $c$ is the center of the ellipsoid, $P$ is the precision matrix, and $x$ is the world coordinate. 

Note that $ P $ is a positive definite matrix, which means that it can be determined by its lower triangular Cholesky decomposition $ P = L^T L $, where $ L $ is the lower triangular matrix. We will use the 6 parameters of $ L $ to parameterize the ellipsoid:

$$
L = \begin{bmatrix}
l_{11} & 0 & 0 \\
l_{21} & l_{22} & 0 \\
l_{31} & l_{32} & l_{33}
\end{bmatrix}
$$

The precision matrix $ P $ is then:

$$
P = L^T L = \begin{bmatrix}
l_{11}^2 & l_{11} l_{21} & l_{11} l_{31} \\
l_{11} l_{21} & l_{21}^2 + l_{22}^2 & l_{21} l_{31} + l_{22} l_{32} \\
l_{11} l_{31} & l_{21} l_{31} + l_{22} l_{32} & l_{31}^2 + l_{32}^2 + l_{33}^2
\end{bmatrix}
$$

Thus, the neural network output layer will consist of the 6 parameters of $ L $:

$$
(l_{11}, l_{21}, l_{31}, l_{22}, l_{32}, l_{33})
$$

These 6 parameters will ensure that $ P $ remains symmetric and positive definite.

### Data generation

Given tha maximum range of each parameter and the image size and the maximum depth, one data point is generated as follows:
1. Sample a parameter uncertainty set (defined by 20-tuple) and a pixel coordinate and depth from the uniform distribution.
2. For each pixel coordinate and depth, and each set of parameters, we will generate $M_S$ samples of world coordinates using monte carlo simulation.


    Each monte carlo sample corresponds to a set of intrinsic and extrinsic parameters sampled from the parameter uncertainty set. Given a camera intrinsic parameters $ \{f_x, f_y, c_x, c_y\} $ and extrinsic parameters $ \{r_x, r_y, r_z, t_x, t_y, t_z\} $, pixel coordinates $ (u, v) $, and depth $ d $, the world coordinates $ (X, Y, Z) $ are calculated as:

$$
\begin{bmatrix}
X \\
Y \\
Z
\end{bmatrix}
= R(\mathbf{r}) \cdot 
\begin{bmatrix}
\frac{(u - c_x) \cdot d}{f_x} \\
\frac{(v - c_y) \cdot d}{f_y} \\
d
\end{bmatrix}
+ 
\begin{bmatrix}
t_x \\
t_y \\
t_z
\end{bmatrix}
$$

Where:
- $ R(\mathbf{r}) $ is the rotation matrix computed from the rotation vector $ \mathbf{r} = [r_x, r_y, r_z] $.
- $ t_x, t_y, t_z $ are the translation parameters.


We will generate $N$ data points using the above procedure.

The data will be stored in a pt file with the following structure:

```python
{
    'parameters': torch.Tensor, # (N, 23) => the intrinsic and extrinsic parameter ranges 
    'pixel': torch.Tensor, # (N, 2) => (u,v)
    'depth': torch.Tensor, # (N, 1)  => d
    'world': torch.Tensor, # (N, M_S, 3) => M_S * (x,y,z)
}
```
where `N` is the batch size and `M_S` is the number of samples generated by the monte carlo simulation.




### Model

We will train a neural network to predict the 9 parameters of the ellipsoid given the pixel coordinate, depth, and the parameters of the camera.

The neural network will have the following architecture:

1. Input layer with 23 neurons 
    ```python
    inputs = (
        u, v, d, 
        f_x_min, f_x_max, f_y_min, f_y_max, 
        c_x_min, c_x_max, c_y_min, c_y_max, 
        rx_min, rx_max, ry_min, ry_max, rz_min, rz_max, 
        tx_min, tx_max, ty_min, ty_max, tz_min, tz_max
        )
    ```
2. 3 hidden layers with 64 neurons each
3. Output layer with 9 neurons (3 for the center and 6 for the Cholesky decomposition of the precision matrix)
    ```python
    outputs = (
    c_x, c_y, c_z,  # 3 for the center of the ellipsoid
    l_11, l_21, l_31, l_22, l_32, l_33  # 6 for the lower triangular elements of the precision matrix
    )
    ```

### Loss function

The loss function is composed of 3 terms: center loss, containment loss, and regularization loss.

#### Center loss

We consider the true center of the ellipsoid to be the mean of the world coordinates generated by the monte carlo simulation. The center loss is the mean squared error between the predicted center and the true center.

$$
\mathcal{L}_{\text{center}} = \frac{1}{N} \sum_{i=1}^{N} (c_i - \hat{c}_i)^2 \\
c_i = \frac{1}{M_S} \sum_{j=1}^{M_S} x_{ij} \\
$$

#### Containment loss

The containment loss is the mean distance between the true world coordinates outside the ellipsoid and the ellipsoid surface.

$$
\mathcal{L}_{\text{containment}} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{M_S} \sum_{j=1}^{M_S} \max(0, (x_{ij} - \hat{c}_i)^T \hat{P}_i (x_{ij} - \hat{c}_i) - 1)\\
or\\
\mathcal{L}_{\text{containment}} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{M_S} \sum_{j=1}^{M_S} \mathbb{1}((x_{ij} - \hat{c}_i)^T \hat{P}_i (x_{ij} - \hat{c}_i) - 1)
$$

#### Regularization loss

The ellipsoid must not be too large, so we add a regularization term to the loss function.

$$
\text{vol}_i = \frac{4}{3} \pi /\sqrt{\text{det}(P_i)} = \frac{4}{3} \pi /|\text{det}(L)|\\
\text{det}(L) = l_{11} \cdot (l_{22} \cdot l_{33} - l_{32} \cdot l_{32}) \\- l_{21} \cdot (l_{21} \cdot l_{33}  - l_{31} \cdot l_{32}) \\+ l_{31} \cdot (l_{21} \cdot l_{32} - l_{22} \cdot l_{31})\\
\mathcal{L}_\text{regularization} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{|\text{det}(L_i)|}
$$


The total loss is given by:

$$  
\mathcal{L} = \lambda_{\text{center}}\mathcal{L}_{\text{center}} + \lambda_{\text{containment}}\mathcal{L}_{\text{containment}} + \lambda_{\text{reg}} \mathcal{L}_{\text{regularization}}
$$

where $\lambda$ is a hyperparameter that controls the importance of the regularization term.

Note that the containment loss and regularization loss must be carefully balanced to ensure that the ellipsoid is not too small or too large.

### Evaluation

We will evaluate the model by computing the containment rate, which is the percentage of true world coordinates that are inside the ellipsoid.

$$
\text{Containment rate} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{M_S} \sum_{j=1}^{M_S} \mathbb{1}((x_{ij} - \hat{c}_i)^T \hat{P}_i (x_{ij} - \hat{c}_i) \leq 1)
$$

