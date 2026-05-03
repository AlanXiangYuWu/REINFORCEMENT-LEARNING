# Sequential Multi-Target Delivery (POMDP)

CA6126 final project. A delivery robot must visit `K` landmarks in
a hidden order that is randomized at every episode. The agent only
gets binary feedback after each `Visit` — so it has to *infer*
the ordering from history.

- **PRD / task outline:** [`docs/PRD.md`](docs/PRD.md)
- **Run-on-server playbook:** [`RUN.md`](RUN.md)
- **Environment:** [`env.py`](env.py) (gymnasium-style)
- **Renderer:** [`render.py`](render.py) (pygame; headless-safe)
- **Trainers:** [`agents/`](agents)
  - `train_ppo.py`, `train_recurrent_ppo.py`, `train_dqn.py`, `train_qrdqn.py`
  - `eval.py`, `record_video.py`, `plot_curves.py`
- **Final report:** `report/report.tex` (written after training)

## Quick local check

```bash
pip install -r requirements.txt
python -m tests.test_env          # gymnasium check_env + rollout sanity
python env.py                     # one random rollout, prints total reward
```

For the full training + evaluation + video pipeline on a GPU box,
follow [`RUN.md`](RUN.md).
