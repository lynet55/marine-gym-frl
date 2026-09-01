# Setting Up MarineGym

Everything in this repo sits on top of
[MarineGym](https://marine-gym.com/) (docs: https://marinegym.netlify.app/,
paper: arXiv:2503.09203) — a research framework for underwater-robot RL in
Isaac Sim. It's not bundled here on purpose; it's a big, shared dependency,
and this doc walks you through getting it (and everything it needs) set up
from scratch, so you're not piecing it together from the upstream README
yourself. See the root [README.md](../README.md) for what this repo
actually imports from it, and [UNDERWATER_PHYSICS.md](UNDERWATER_PHYSICS.md)
if you're curious what MarineGym is doing physics-wise once it's running.

## First, a heads-up: you can't clone just MarineGym on its own

I checked this directly against the source checkout this repo was pulled
from: `marinegym/` has no independent git remote, and no `setup.py` or
`pyproject.toml` of its own. It only exists as a subdirectory inside the
larger MarineGym monorepo — the one with the top-level `setup.py` that
actually registers it as an installable package. So you need that whole
monorepo checkout, not just the `marinegym/` folder. If you don't already
have one sitting around, start from the install guides linked at
https://marinegym.netlify.app/, or the project's own README. I'm not going
to hand you a clone URL here, because this repo's extraction predates a
confirmed public one — the docs site above is the authoritative place to
find wherever the source lives right now.

## 1. Isaac Sim

Install Isaac Sim from the Omniverse launcher or a local tarball. MarineGym's
own README documents two paths:

- Isaac Sim **4.1.0** — the original badge/target version, or
- Isaac Sim **5.1** + IsaacLab **v2.3.2** — the combination MarineGym's
  README specifically calls out as tested end-to-end, and what this repo
  was actually verified against (`ISAAC_PY`'s default here points at an
  Isaac Sim 5.1 `python.sh`, for what it's worth).

Python 3.10+ comes bundled with Isaac Sim, so there's nothing to install
separately there. No CUDA/driver version is pinned anywhere in the upstream
docs — just use whatever Isaac Sim's own installer asks for on your GPU. You
do need an NVIDIA GPU; both the physics and the rendering are PhysX/RTX
based, there's no CPU fallback path here.

If you're going the Isaac Sim 5.1 + IsaacLab route:

```bash
cd /path/to/your/workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2

ln -sfn /path/to/isaac-sim-5.1 _isaac_sim
./_isaac_sim/python.sh -m pip install -U pip
./_isaac_sim/python.sh -m pip install -e .
./_isaac_sim/python.sh -m pip install -e source/isaaclab_rl
```

## 2. MarineGym itself

Once Isaac Sim's in place, from inside your MarineGym monorepo checkout
(the one with `marinegym/` and its top-level `setup.py` sitting right next
to each other):

```bash
cd /path/to/MarineGym-main
<isaac-sim>/python.sh -m pip install "tensordict==0.7.2" "torchrl==0.7.2"
<isaac-sim>/python.sh -m pip install -e .
```

This installs `marinegym` as an editable package straight into Isaac Sim's
bundled Python environment — no conda env, no separate virtualenv needed,
it's all riding on Isaac Sim's own interpreter. It'll also pull in
MarineGym's other Python dependencies along the way: `hydra-core`,
`omegaconf`, `wandb`, `imageio`, `plotly`, `einops`, `pandas`, `moviepy`,
`av`, `setproctitle`.

Once that's done, a quick sanity check:

```bash
<isaac-sim>/python.sh -c "import marinegym; print(marinegym.__file__)"
```

You want to see a path ending in `marinegym/__init__.py` with no errors. If
you get an import error here, don't bother going further — fix it first,
because every script in this repo depends on it.

## 3. Pointing this repo at your Isaac Sim + MarineGym

There's nothing to install *in this repo itself* — it just needs
`marinegym` to be importable from whichever Python you launch with:

```bash
export ISAAC_PY=/path/to/isaac-sim/python.sh   # defaults to /home/eth11/isaacsim51/python.sh
```

One thing worth knowing: the tank scene is already bundled right here in
this repo (`assets/usd/BlueROV/tank.usd`, fully self-contained — see
[CUSTOM_ENVIRONMENT.md](CUSTOM_ENVIRONMENT.md)), so you don't need
MarineGym for that part. The BlueROV **robot** model, though
(`BlueROV.usd`/`BlueROV.yaml`), is resolved internally by MarineGym from
your checkout — so a working MarineGym install is required for literally
any run here, even the simplest environment-only sanity check.

## 4. ROS 2 Jazzy

This one's unrelated to MarineGym — you only need it for the ROS2
bridges and RViz visualization pieces of this repo, not for MarineGym
itself:

```bash
source /opt/ros/jazzy/setup.bash
```

If you don't have Jazzy installed yet, follow ROS 2's own install docs for
your distro: https://docs.ros.org/en/jazzy/Installation.html

## 5. Checking it all actually works together

Once everything above is in place, this is the fastest way to confirm it:

```bash
cd bluerov-bt-autonomous
HEADLESS=true ACTION_MODE=hold NUM_ENVS=1 STEPS=50 CLOSE_ON_FINISH=true \
ROS2_TELEMETRY_ENABLE=false \
bash release/scripts/00_open_hover_vel_environment.sh
```

If everything's wired up right, you'll see Isaac Sim boot, a
`BLUEROV HOVER_VEL ENV DEBUG` banner print, 50 steps run, and the whole
thing exit cleanly with `finished stepping in ...s` — no
`ModuleNotFoundError: No module named 'marinegym'`, no traceback. If you do
see that import error specifically, it almost always means step 2's
`pip install -e .` targeted a different Python than the one you're actually
launching with in `ISAAC_PY` — go back and check they match.
