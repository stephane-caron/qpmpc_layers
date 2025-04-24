import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import qpmpc_layers.lti as lti
import qpmpc_layers

# Constants
DT = 0.1
NB_TIMESTEPS = 50
NX, NU = 2, 1


def get_dynamics_matrices():
    """Returns A and B matrices for a simple double integrator system."""
    A = torch.tensor([[1, DT], [0, 1]], dtype=torch.float32)
    B = torch.tensor([[0], [DT]], dtype=torch.float32)
    return A, B


class LearnableMatrix(nn.Module):
    """A learnable matrix module."""
    def __init__(self, matrix):
        super().__init__()
        self.matrix = nn.Parameter(matrix)

    def forward(self):
        return self.matrix


class ConstraintModule(nn.Module):
    """Constraint module wrapper for learnable constraints."""
    def __init__(self, C, D, e, f):
        super().__init__()
        self.C = C
        self.D = D
        self.e = e
        self.f = f

    def forward(self):
        return qpmpc_layers.Constraint(
            C=self.C(),
            D=self.D(),
            e=self.e(),
            f=self.f(),
        )


def plot_results(x, v, a, filename="plot.png"):
    """Plots position, velocity, and acceleration over time."""
    fig, ax = plt.subplots(3, 1, figsize=(10, 6))

    labels = ["Position (m)", "Velocity (m/s)", "Acceleration (m/s²)"]
    datas = [x, v, a]
    titles = ["Position", "Velocity", "Acceleration"]

    for i in range(3):
        ax[i].plot(datas[i].numpy(), label=labels[i].split()[0])
        ax[i].set_title(titles[i])
        ax[i].set_xlabel("Time step")
        ax[i].set_ylabel(labels[i])
        ax[i].legend()

    plt.tight_layout()
    plt.savefig(filename)


def create_ocp(A, B, constraint):
    """Creates an OptimalControlProblem using provided dynamics and constraint."""
    dynamics = qpmpc_layers.Dynamics(A=A, B=B)
    x_goal = torch.tensor([10.0, 0.0])
    stage_cost = qpmpc_layers.StageCost(
        Q=torch.eye(NX),
        q=-x_goal,
        R=torch.zeros((NU, NU)),
    )
    return lti.OptimalControlProblem(
        nb_steps=NB_TIMESTEPS,
        stage_cost=stage_cost,
        dynamics=dynamics,
        constraint=constraint
    )


if __name__ == "__main__":
    A, B = get_dynamics_matrices()

    # True constraint (velocity and acceleration bounds)
    v_max, a_max = 10.0, 1.0
    true_constraint = qpmpc_layers.Constraint(
        C=torch.tensor([
            [0.0, 1.0],  # +v ≤ v_max
            [0.0, 0.0],  # a constraint (in D)
        ]),
        D=torch.tensor([
            [0.0],
            [1.0],
        ]),
        e=torch.tensor([
            -v_max,
            -a_max,
        ]),
        f=torch.tensor([
            v_max,
            a_max,
        ])
    )

    # Initial guess for the learnable constraint
    learned_constraint_module = ConstraintModule(
        C=LearnableMatrix(torch.zeros_like(true_constraint.C)),
        D=LearnableMatrix(torch.zeros_like(true_constraint.D)),
        e=LearnableMatrix(-torch.ones_like(true_constraint.e)),
        f=LearnableMatrix(torch.ones_like(true_constraint.f)),
    )

    # Define OCPs
    true_ocp = create_ocp(A, B, true_constraint)
    learned_ocp = create_ocp(A, B, learned_constraint_module)

    qplayer = qpmpc_layers.QPLayer(max_iter=10)
    optimizer = torch.optim.Adam(learned_ocp.constraint.parameters(), lr=1e-2, weight_decay=1e-4)

    for iteration in range(1000):
        x_init = 20*torch.rand(2)-10
        with torch.no_grad():
            result = qplayer(true_ocp(x_init)).flatten()
        x_terminal = result[-NX:]
        result = result[:-NX].reshape(NB_TIMESTEPS, NX + NU)
        inputs_star = result[:, -NU:]
        states_star = torch.cat([result[:, :-NU], x_terminal.unsqueeze(0)])

        learned_constraint = learned_ocp.constraint()
        result = qplayer(learned_ocp(x_init)).flatten()

        x_terminal = result[-NX:]
        result = result[:-NX].reshape(NB_TIMESTEPS, NX + NU)
        inputs = result[:, -NU:]
        states = torch.cat([result[:, :-NU], x_terminal.unsqueeze(0)])

        reg_term = (
            learned_constraint.C.norm() +
            learned_constraint.D.norm() +
            learned_constraint.e.norm() +
            learned_constraint.f.norm()
        )
        loss = torch.mean((inputs - inputs_star) ** 2) + torch.mean((states - states_star) ** 2)
        print(f"Iter {iteration:4d} | Loss: {loss.item():.4f} | Reg: {reg_term.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learned_ocp.constraint.parameters(), 0.5)
        optimizer.step()

    # Final rollout and plot
    with torch.no_grad():
        final_result = qplayer(learned_ocp(x_init)).flatten()
    x_terminal = final_result[-NX:]
    final_result = final_result[:-NX].reshape(NB_TIMESTEPS, NX + NU)
    inputs = final_result[:, -NU:]
    states = torch.cat([final_result[:, :-NU], x_terminal.unsqueeze(0)])
    x, v, a = states[:, 0], states[:, 1], inputs[:, 0]
    plot_results(x.cpu(), v.cpu(), a.cpu())

    # Print final learned vs true constraints
    final_constraint = learned_ocp.constraint()
    print("\n=== Constraint Comparison ===")
    print("True C:\n", true_constraint.C)
    print("Learned C:\n", final_constraint.C.detach())
    print("\nTrue D:\n", true_constraint.D)
    print("Learned D:\n", final_constraint.D.detach())
    print("\nTrue e:\n", true_constraint.e)
    print("Learned e:\n", final_constraint.e.detach())
    print("\nTrue f:\n", true_constraint.f)
    print("Learned f:\n", final_constraint.f.detach())
