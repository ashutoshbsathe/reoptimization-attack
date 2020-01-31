# ![formula](https://render.githubusercontent.com/render/math?math=\theta)-space Adversarial Attacks

From the given model parameters ![formula](https://render.githubusercontent.com/render/math?math=\theta), find parameters ![formula](https://render.githubusercontent.com/render/math?math=\theta^*) such that adversaries generated from ![formula](https://render.githubusercontent.com/render/math?math=\theta^*) are stronger than adversaries generated from \(\theta\).

The repo contains target and substitute models. To evaluate a particular training method, run `eval_{training_method_name}.py` file.

## Requirements
```
advertorch
torch
python3
```