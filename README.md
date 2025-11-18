# qpmpc\_layers

Differentiable linear model predictive control in Python, for optimal-control problems that are quadratic programs. This library revisits [qpmpc](https://github.com/stephane-caron/qpmpc) with a new API better suited to [PyTorch](https://pytorch.org/), and builds upon [QPLayer](https://github.com/Simple-Robotics/proxsuite?tab=readme-ov-file#qplayer) to solve QPs as part of PyTorch computation graphs.

> [!WARNING]
> qpmc\_layers is at the alpha stage of its development.

Expect breaking changes and missing features, but feel also free to join the discussion or add work-in-progress examples at this stage.

## Development

- Checking unit tests: ``pixi run -e py310 test``
- Building and opening the documentation locally: ``pixi run docs-open``
