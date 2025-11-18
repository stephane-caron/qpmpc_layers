# qpmpc\_layers

[![CI](https://img.shields.io/github/actions/workflow/status/stephane-caron/qpmpc_layers/ci.yml?branch=main)](https://github.com/stephane-caron/qpmpc_layers/actions)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen?style=flat)](https://stephane-caron.github.io/qpmpc_layers/)
[![Coverage](https://coveralls.io/repos/github/stephane-caron/qpmpc_layers/badge.svg?branch=main)](https://coveralls.io/github/stephane-caron/qpmpc_layers?branch=main)
[![PyPI version](https://img.shields.io/pypi/v/qpmpc_layers)](https://pypi.org/project/qpmpc_layers/0.6.0/)

Differentiable linear model predictive control in Python, for optimal-control problems that are quadratic programs. This library revisits [qpmpc](https://github.com/stephane-caron/qpmpc) with a new API better suited to [PyTorch](https://pytorch.org/), and builds upon [QPLayer](https://github.com/Simple-Robotics/proxsuite?tab=readme-ov-file#qplayer) to solve QPs as part of PyTorch computation graphs.

> [!WARNING]
> qpmc\_layers is still under development, expect breaking changes.

Feel also free to join the discussion or add work-in-progress examples at this stage.

## Development

- Checking unit tests: ``pixi run -e py310 test``
- Building and opening the documentation locally: ``pixi run docs-open``
