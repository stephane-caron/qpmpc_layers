import casadi as ca
import numpy as np

N = 20

A = ca.DM([[0, 1], [-1, -1]])
B = ca.DM([[0], [1]])
Q = ca.DM([[1, 0], [0, 1]])
q = ca.DM([[2], [1]])
R = ca.DM([[1]])
r = ca.DM([[0]])

U = ca.MX.sym("U", N)
x_init = ca.MX.sym("x_init", 2)

cost = 0.0
g = []
x_k_next = x_init
for k in range(N):
    x_k = x_k_next
    x_k_next = A @ x_k + B @ U[k]
    cost += 0.5 * ca.mtimes(x_k.T, ca.mtimes(Q, x_k)) + ca.mtimes(q.T, x_k)
    cost += 0.5 * ca.mtimes(U[k].T, ca.mtimes(R, U[k])) + ca.mtimes(r.T, U[k])
    g.append(x_k_next[0])  # x[0] == 0
    g.append(x_k_next[1])  # x[1] == 0

nlp = {"x": U, "f": cost, "g": ca.vertcat(*g), "p": x_init}
lbg = [0] * len(g)  # lower bound for g
ubg = [0] * len(g)  # upper bound for g

opts = {"ipopt": {"print_level": 0}}
solver = ca.nlpsol("solver", "ipopt", nlp, opts)
U_guess = np.zeros(N)
solution = solver(x0=U_guess, lbg=lbg, ubg=ubg)
U_star = solution["x"]
print(f"Optimal control input: {U_star=}")
