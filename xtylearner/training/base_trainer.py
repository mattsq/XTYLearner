### currently storing various implementations of trainer functions
## TODO: consolidate these into a generic interface to the models/

### M2 VAE

class M2VAE(nn.Module):
    def __init__(self, d_x, d_y, k, d_z=16, tau=0.5):
        super().__init__()
        self.enc_z = EncoderZ(d_x, k, d_z)
        self.cls_t = ClassifierT(d_x, d_y, k)
        self.dec_x = DecoderX(d_z, d_x)
        self.dec_t = DecoderT(d_x, d_z, k)
        self.dec_y = DecoderY(d_x, k, d_z, d_y)
        self.k     = k
        self.tau   = tau                    # Gumbel temperature

    # --------------------------------------------------------
    # one minibatch update (labelled + unlabelled mixed)
    def elbo(self, x, y, t_obs):
        """
        x : (B, d_x),  y : (B, d_y)
        t_obs : (B,)   integer in [0,K-1] or -1 for 'missing'
        """
        B = x.size(0)
        labelled   = (t_obs >= 0)
        unlabelled = ~labelled

        # ---------- labelled branch ----------------------------------
        t_lab      = t_obs[labelled]
        t_onehot_L = one_hot(t_lab, self.k).float()
        z_L, mu_L, logvar_L = self.enc_z(x[labelled], t_onehot_L)

        # likelihood terms
        recon_x_L  = Normal(self.dec_x(z_L), 1.0).log_prob(x[labelled]).sum(-1)
        logits_t_L = self.dec_t(x[labelled], z_L)
        logp_t_L   = -nn.CrossEntropyLoss(reduction='none')(logits_t_L, t_lab)
        recon_y_L  = Normal(self.dec_y(x[labelled], t_onehot_L, z_L), 1.0)\
                     .log_prob(y[labelled]).sum(-1)

        # KL(z|x,t)
        kl_L = -0.5 * (1 + logvar_L - mu_L.pow(2) - logvar_L.exp()).sum(-1)

        elbo_L = (recon_x_L + logp_t_L + recon_y_L - kl_L).mean()

        # ---------- unlabelled branch --------------------------------
        if unlabelled.any():
            # q(t|x,y)
            logits_q  = self.cls_t(x[unlabelled], y[unlabelled])
            q_t       = softmax(logits_q, -1)             # (B_u, K)

            # Gumbel-Softmax sample for reparameterised t
            t_soft    = gumbel_softmax(logits_q, tau=self.tau, hard=False)
            z_U, mu_U, logvar_U = self.enc_z(x[unlabelled], t_soft)

            # expected likelihoods under q(t)
            recon_x_U = Normal(self.dec_x(z_U), 1.0).log_prob(x[unlabelled])\
                        .sum(-1)
            logits_t_U = self.dec_t(x[unlabelled], z_U)
            logp_t_U   = -(q_t * logits_t_U.log_softmax(-1)).sum(-1)
            recon_y_U  = Normal(self.dec_y(x[unlabelled], t_soft, z_U), 1.0)\
                         .log_prob(y[unlabelled]).sum(-1)
            kl_U       = -0.5 * (1 + logvar_U - mu_U.pow(2) - logvar_U.exp())\
                         .sum(-1)

            elbo_U = (recon_x_U + recon_y_U - kl_U + logp_t_U +
                      (- (q_t * q_t.log()).sum(-1))    # −H[q(t|x,y)]
                     ).mean()
        else:
            elbo_U = 0.0

        # supervised CE to keep classifier sharp
        ce_sup = 0.0
        if labelled.any():
            ce_sup = nn.CrossEntropyLoss()(self.cls_t(x[labelled], y[labelled]),
                                            t_lab)

        # maximise ELBO  → minimise −ELBO
        loss = -(elbo_L + elbo_U) + ce_sup
        return loss

# data loaders -------------------------------------------------------------
#  X_train : (N,d_x) float32
#  Y_train : (N,d_y) float32
#  T_train : (N,)    int64  with -1 where treatment is unobserved
batch = 256
loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, Y_train, T_train),
            batch_size=batch, shuffle=True, drop_last=True)

# model + optimiser --------------------------------------------------------
model = M2VAE(d_x=X_train.shape[1], d_y=Y_train.shape[1], k=K).to('cpu')
opt   = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(200):
    for x, y, t in loader:
        opt.zero_grad()
        loss = model.elbo(x, y, t)
        loss.backward()
        opt.step()
    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.3f}")


# 1. p_theta(Y | X, choose treatment t*)
with torch.no_grad():
    z    = torch.randn_like(torch.empty(X_new.size(0), model.enc_z.net_mu[-1].out_features))
    t_1h = one_hot(torch.full((X_new.size(0),), t_star), model.k).float()
    y_mu = model.dec_y(X_new, t_1h, z)      # predicted mean
    y_sd = torch.ones_like(y_mu)            # unit variance in this toy set-up

# 2. Infer T from (x,y)
probs_T = softmax(model.cls_t(X_new, Y_new), -1)      # q_phi(t|x,y)

# 3. Reconstruct plausible X given (t,y)
#    sample z ∼ N(0,1), then x̃ = dec_x(z)

### semi-supervised CEVAE

class SS_CEVAE(nn.Module):
    def __init__(self, d_x, d_y, k=2, d_z=16, tau=0.5):
        super().__init__()
        self.enc_z = EncoderZ(d_x, k, d_y, d_z)
        self.cls_t = ClassifierT(d_x, d_y, k)
        self.dec_x = DecoderX(d_z, d_x)
        self.dec_t = DecoderT(d_z, d_x, k)
        self.dec_y = DecoderY(d_z, d_x, k, d_y)
        self.k = k
        self.tau = tau                        # Gumbel-Softmax temp

    # ------------------------------------------------------------------
    def elbo(self, x, y, t_obs):
        """
        x: (B,d_x)  y: (B,d_y)
        t_obs: (B,)  int in [0,K-1] or −1 for missing
        """
        B = x.size(0)
        lab = t_obs >= 0
        unlab = ~lab

        # -------- labelled branch -------------------------------------
        t_lab = t_obs[lab]
        t1h_L = one_hot(t_lab, self.k).float()
        z_L, mu_L, logv_L = self.enc_z(x[lab], t1h_L, y[lab])

        log_px_L = Normal(self.dec_x(z_L), 1.).log_prob(x[lab]).sum(-1)
        log_pt_L = -nn.CrossEntropyLoss(reduction='none')(
                       self.dec_t(z_L, x[lab]), t_lab)
        log_py_L = Normal(self.dec_y(z_L, x[lab], t1h_L), 1.)\
                       .log_prob(y[lab]).sum(-1)
        kl_L = -0.5*(1 + logv_L - mu_L.pow(2) - logv_L.exp()).sum(-1)
        elbo_L = (log_px_L + log_pt_L + log_py_L - kl_L).mean()

        # -------- unlabelled branch -----------------------------------
        elbo_U = torch.tensor(0., device=x.device)
        if unlab.any():
            logits_q = self.cls_t(x[unlab], y[unlab])
            q_t = softmax(logits_q, -1)               # (B_u,K)
            t_soft = gumbel_softmax(logits_q, tau=self.tau, hard=False)
            z_U, mu_U, logv_U = self.enc_z(x[unlab], t_soft, y[unlab])

            log_px_U = Normal(self.dec_x(z_U), 1.).log_prob(x[unlab]).sum(-1)
            logits_pT = self.dec_t(z_U, x[unlab])
            log_pt_U = -(q_t * logits_pT.log_softmax(-1)).sum(-1)
            log_py_U = Normal(self.dec_y(z_U, x[unlab], t_soft), 1.)\
                           .log_prob(y[unlab]).sum(-1)
            kl_U = -0.5*(1 + logv_U - mu_U.pow(2) - logv_U.exp()).sum(-1)
            H_q = -(q_t * q_t.log()).sum(-1)

            elbo_U = (log_px_U + log_pt_U + log_py_U - kl_U + H_q).mean()

        # supervised CE to keep classifier sharp
        ce_sup = 0.
        if lab.any():
            ce_sup = nn.CrossEntropyLoss()(
                         self.cls_t(x[lab], y[lab]), t_lab)

        loss = -(elbo_L + elbo_U) + ce_sup
        return loss

# X_train : (N,d_x)  Y_train : (N,d_y)
# T_train : (N,) int64 with −1 where treatment is unobserved
loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, Y_train, T_train),
            batch_size=256, shuffle=True, drop_last=True)

model = SS_CEVAE(d_x=X_train.shape[1],
                 d_y=Y_train.shape[1],
                 k=num_treatments).to('cpu')
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

for ep in range(200):
    for x_b,y_b,t_b in loader:
        opt.zero_grad(); loss = model.elbo(x_b,y_b,t_b); loss.backward(); opt.step()
    if ep % 20 == 0:
        print(f"epoch {ep:3d}  loss {loss.item():.3f}")

# 1. Predict Y distribution under a chosen treatment t*
with torch.no_grad():
    z = torch.randn(N_test, 16)                # prior draw
    t1h = one_hot(torch.full((N_test,), t_star), model.k).float()
    y_mu = model.dec_y(z, X_test, t1h)         # mean, var=1
# 2. Infer treatment from (x,y)
probs_T = softmax(model.cls_t(X_new, Y_new), -1)
# 3. Draw plausible X from (t,y):  sample z| … then x = dec_x(z)


### Flow SSC

# X_train : (N,d_x) float32
# Y_train : (N,d_y) float32
# T_train : (N,) int64 with −1 where treatment unobserved
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, Y_train, T_train),
    batch_size=256, shuffle=True, drop_last=True)

model = MixtureOfFlows(dim_x=X_train.shape[1],
                       dim_y=Y_train.shape[1],
                       n_treat=K).to('cpu')
opt = torch.optim.Adam(model.parameters(), lr=2e-4)

for epoch in range(150):
    for xb, yb, tb in loader:
        opt.zero_grad()
        loss = model.loss(xb, yb, tb)
        loss.backward()
        opt.step()
    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.3f}")


## Cycle-Dual

#  X_train : (N,d_x) float32
#  Y_train : (N,d_y) float32
#  T_train : (N,)    int64  (−1 where missing)
batch = 256
loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, Y_train, T_train),
            batch_size=batch, shuffle=True, drop_last=True)

model = CycleDual(d_x=X_train.shape[1],
                  d_y=Y_train.shape[1],
                  k=num_treatments).to('cpu')
opt   = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(300):
    for xb, yb, tb in loader:
        opt.zero_grad()
        loss = model.loss(xb, yb, tb,
                          λ_sup=1.0, λ_cyc=1.0, λ_ent=0.1)
        loss.backward()
        opt.step()
    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

## Multitask Selftrain

def train_self_supervised(model, X, Y, T_obs,
                          k, n_outer=10, n_inner=3,
                          batch=256, thr=0.9, lr=1e-3):
    """
    n_outer : pseudo-labelling rounds
    n_inner : SGD epochs per round
    thr     : confidence threshold for accepting pseudo labels
    """

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # masks that say which rows are treated as "labelled" right now
    labelled = (T_obs >= 0).clone()
    T_curr   = T_obs.clone()          # will hold pseudo labels

    for outer in range(n_outer):
        loader = torch.utils.data.DataLoader(
                     DataWrapper(X, Y, T_curr),
                     batch_size=batch, shuffle=True, drop_last=True)

        # ------- train on current label set ---------------------------
        for _ in range(n_inner):
            for xb, yb, tb in loader:
                T_known = tb >= 0
                if not T_known.any():     # skip batch with no labels
                    continue

                T_1h = one_hot(tb.clamp(min=0), k).float()   # −1→0
                Y_hat, logits_T, X_hat = model(xb, yb, T_1h)

                # losses only on rows where T is (true or pseudo) known
                idx = T_known.nonzero(as_tuple=True)[0]
                L_Y = mse_loss(Y_hat[idx], yb[idx])
                L_X = mse_loss(X_hat[idx], xb[idx])
                L_T = cross_entropy(logits_T[idx], tb[idx])

                loss = L_Y + L_X + L_T
                opt.zero_grad(); loss.backward(); opt.step()

        # ------- pseudo-labelling step --------------------------------
        with torch.no_grad():
            logits_full = model.head_T(torch.cat([X, Y], -1))
            probs = logits_full.softmax(-1)
            conf, pred = probs.max(-1)
            new_pseudo = (labelled == 0) & (conf > thr)

            # update masks and label tensor
            T_curr[new_pseudo] = pred[new_pseudo]
            labelled |= new_pseudo

        print(f"round {outer+1:2d}/{n_outer}  "
              f"labelled now: {labelled.float().mean()*100:.1f}%")

    return model

# tensors:  X_all (N,d_x)  Y_all (N,d_y)  T_all (N,)  with −1 where unknown
K = int(T_all[T_all>=0].max()) + 1

model = MultiTask(d_x=X_all.shape[1],
                  d_y=Y_all.shape[1],
                  k=K)

model = train_self_supervised(model, X_all, Y_all, T_all,
                              k=K, n_outer=8, n_inner=2)
# 1. Predict outcome for a chosen treatment t*
# x_test = ...
t_star = 2
y_pred = model.head_Y(
            torch.cat([model.h(x_test),
                       one_hot(torch.tensor([t_star]), K).float()], -1))

# 2. Infer treatment from (x,y)
probs_t = model.head_T(torch.cat([x_test, y_real], -1)).softmax(-1)

# 3. Reconstruct plausible X given (t,y)
x_hat = model.head_X(torch.cat([y_real,
                                one_hot(torch.tensor([t_val]),K).float()], -1))
