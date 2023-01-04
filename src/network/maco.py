def get_mapper(n_in, n_h1, n_h2, n_out):
    mapper = Sequential(Linear(n_in, n_h1),
                        ReLU(),
                        Linear(n_h1, n_h2),
                        ReLU(),
                        Linear(n_h2, n_out))
    return mapper


def get_coach(n_in, n_h1, n_out):
    coach = Sequential(Linear(n_in, n_h1),
                       ReLU(),
                       Linear(n_h1, n_out))
    return coach


class MaCo(torch.nn.Module):
    def __init__(self, Ex, Ey, Ez, mh_kwargs, ch_kwargs, device):
        super().__init__()
        self.mapper = get_mapper(n_in=Ey, n_out=Ez, **mh_kwargs)
        self.coach_x = get_coach(Ex + Ez, n_out=1, **ch_kwargs)

        self.Ex = Ex
        self.Ey = Ey
        self.Ez = Ez
        self.mh_params = mh_kwargs
        self.ch_params = ch_kwargs
        self.train_loss_history = []
        self.criterion = MSELoss()
        self.device = device

    def forward(self, q):
        z = self.mapper.forward(q[:, self.Ex:self.Ex + self.Ey])
        hz = relu(z)

        mx = torch.cat((hz, q[:, :self.Ex]), axis=1)

        pred = self.coach_x.forward(mx)
        return pred, z, hz

    def train_loop(self, loader, n_epochs, lr=1e-2):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        loss_hist = self.train_loss_history.copy()
        for epoch in tqdm(range(n_epochs), leave=False):
            losses = []
            for i, batch in enumerate(loader):
                q, target = batch

                self.optimizer.zero_grad()

                pred, z, hz = self.forward(q.to(self.device))

                loss = self.criterion(target.to(self.device), pred)

                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
            loss_hist.append(np.mean(losses))
        self.train_loss_history = loss_hist.copy()
        return loss_hist

    def test_loop(self, loader):
        q, target = loader
        pred, z, hz = self.forward(q)
        loss = self.criterion(target, pred).item()
        return loss

    def valid_loop(self, loader):
        q, target = loader
        pred, z, hz = self.forward(q)
        loss = self.criterion(target, pred).item()
        return loss, pred.squeeze().detach().numpy(), z.squeeze().detach().numpy(), hz.squeeze().detach().numpy()
