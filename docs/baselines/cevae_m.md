CEVAE-M extends the original CEVAE to settings where the treatment label is missing for many samples. It keeps the CEVAE generative structure but models missing treatments as latent variables.

### Objective

For each mini-batch the model samples a soft treatment assignment `t` from `q(t|x,y)` using the Gumbel-Softmax trick. The latent confounder `z` is drawn from `q(z|x,y,t)`. The evidence lower bound is

\[
\mathbb E_{q(t|x,y)}\Big[\mathbb E_{q(z|x,y,t)}[\log p(x|z)+\log p(t|z)+\log p(y|x,t,z)]-\mathrm{KL}(q(z|x,y,t)\,||\,p(z))\Big]
\]
plus a cross-entropy term on observed treatments.

### Tips

- Temperature `tau` controls the sharpness of the Gumbel-Softmax samples.
- A larger latent dimension `d_z` often helps on IHDP style data.
- Train for roughly 400 epochs with Adam at learning rate 1e-3.

### Citation

```
@inproceedings{louizos2017cevae,
  title     = {Causal Effect Inference with Deep Latent-Variable Models},
  author    = {Louizos, Christos and others},
  booktitle = {NeurIPS},
  year      = {2017}
}
```
