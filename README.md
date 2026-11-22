# Learning AD

A small project to learn the basics of automatic differentiation (AD): the theory, how it works under the hood, and how to use it in practice in Julia and in JAX.

## Contents

### 1. Theoretical development
`Notes_AD.pdf`: notes on the theory behind AD — forward and reverse mode, dual numbers, JVPs and VJPs.

### 2. Basics of AD in Julia
`AD_notebook.ipynb`
1. Forward-mode AD coded from scratch with dual numbers and operator overloading
2. AD libraries: ForwardDiff (forward mode) and Zygote (reverse mode)
3. JVP vs VJP, and building Jacobians column by column or row by row
4. Forward-over-reverse: Hessians, Hessian–vector products and scalar functions of gradients
5. An example of implicit differentiation: the derivative of a fixed point, in the spirit of DFTK's polarizability calculation

### 3. JAX in context: a hike
`Hike_with_JAX.ipynb` (plotting helpers in `hike.py`)

A hiker explores a landscape in the fog, and each JAX tool is introduced when the hiker needs it:
- **The compass:** `jax.grad`, checked against finite differences
- **Steepest ascent:** line searches with Newton's method, nested `grad`s, and `jax.hessian` to confirm the summit
- **The lowest point on a trail:** constrained minimisation with Lagrange multipliers and `jax.jacobian`

Along the way: `vmap` for evaluating over many points at once, and interactive 3D plots with Plotly.

## Running the notebooks

**Julia** (1.13): from the repository folder,
```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
IJulia is needed to run the notebook in Jupyter.

**Python** (≥ 3.12, environment managed with [uv](https://docs.astral.sh/uv/)):
```
uv sync
uv run python -m ipykernel install --user --name learning-ad-jax --display-name "Python (JAX, Learning_AD)"
uv run jupyter notebook
```
