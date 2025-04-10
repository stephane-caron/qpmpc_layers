
import torch
class QPResult:
    def __init__(self, flat_result, nx, nu):
        if flat_result.ndim ==1 :
            flat_result = flat_result.unsqueeze(0)
        self.n_batch = flat_result.shape[0]
        self.flat_result = flat_result
        self.nx = nx
        self.nu = nu
        self.trajectory, self.inputs = self.process_result()

    def process_result(self):
        last_x = self.flat_result[..., -self.nx:]
        self.flat_result = self.flat_result[..., :-self.nx]
        step_size = self.nx + self.nu
        N = self.flat_result.shape[1] // step_size

        # Reshape into (N, nx + nu)
        reshaped = self.flat_result.view(self.n_batch,N, step_size)

        # Split into trajectory and inputs
        trajectory = reshaped[:,:, :self.nx]
        trajectory = torch.cat((trajectory, last_x.unsqueeze(1)), dim=1)
        inputs = reshaped[:,:, self.nx:]

        return trajectory, inputs
