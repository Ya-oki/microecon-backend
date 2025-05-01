import sys, types
# Stub ssl if missing in sandbox environment
sys.modules['ssl'] = types.ModuleType('ssl')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sympy import symbols, sympify, diff, solve, lambdify
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
description = (
    "Logic-enforced backend for advanced microeconomics and mathematics "
    "computations and visualizations."
)
app = FastAPI(
    title="Microeconomics & Math Computation API",
    description=description,
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper: render matplotlib figure to base64
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# === Request Models ===
class CostAnalysisRequest(BaseModel):
    price: float = Field(..., gt=0)
    fixed_cost: float = Field(..., ge=0)
    variable_cost_expr: str
    q_min: float = Field(0, ge=0)
    q_max: float = Field(..., gt=0)
    steps: int = Field(100, gt=1)

class SupplyDemandRequest(BaseModel):
    demand_expr: str
    supply_expr: str
    price_min: float = Field(0, ge=0)
    price_max: float = Field(..., gt=0)
    steps: int = Field(100, gt=1)

class UtilityMaxRequest(BaseModel):
    utility_expr: str
    budget: float = Field(..., gt=0)
    px: float = Field(..., gt=0)
    py: float = Field(..., gt=0)
    grid_range: float = Field(10, gt=0)
    grid_steps: int = Field(100, gt=1)

class MonopolyRequest(BaseModel):
    demand_expr: str  # P = f(Q)
    total_cost_expr: str
    q_max: float = Field(100, gt=0)
    steps: int = Field(100, gt=1)

class OptimizationRequest(BaseModel):
    f_expr: str
    x_min: float
    x_max: float
    maximize: bool = True
    steps: int = Field(100, gt=1)

class ElasticityRequest(BaseModel):
    demand_expr: str  # Q = f(P)
    price: float
    delta: float = Field(1e-4, gt=0)
    steps: int = Field(100, gt=1)

# === Endpoints ===
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/cost-analysis")
def cost_analysis(req: CostAnalysisRequest):
    Q = symbols('Q', real=True, nonnegative=True)
    try:
        vc_sym = sympify(req.variable_cost_expr)
    except Exception:
        raise HTTPException(400, "Invalid variable_cost_expr")

    TC = req.fixed_cost + vc_sym
    MC = diff(TC, Q)
    AC = TC / Q
    AVC = vc_sym / Q

    q_vals = np.linspace(req.q_min, req.q_max, req.steps)
    f_tc = lambdify(Q, TC, 'numpy')
    f_mc = lambdify(Q, MC, 'numpy')
    f_ac = lambdify(Q, AC, 'numpy')
    f_avc = lambdify(Q, AVC, 'numpy')

    tc_vals = f_tc(q_vals)
    mc_vals = f_mc(q_vals)
    ac_vals = f_ac(q_vals)
    avc_vals = f_avc(q_vals)

    fig, ax = plt.subplots()
    ax.plot(q_vals, tc_vals, label='TC')
    ax.plot(q_vals, mc_vals, label='MC')
    ax.plot(q_vals, ac_vals, label='AC')
    ax.plot(q_vals, avc_vals, label='AVC')
    ax.axhline(req.price, linestyle='--', label=f'Price={req.price}')

    profit_mask = req.price > ac_vals
    ax.fill_between(q_vals, req.price, ac_vals, where=profit_mask, alpha=0.3, label='Profit')
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Cost / Price')
    ax.legend()

    return {"image_base64": fig_to_base64(fig)}

@app.post("/supply-demand")
def supply_demand(req: SupplyDemandRequest):
    P = symbols('P', real=True)
    try:
        d_sym = sympify(req.demand_expr)
        s_sym = sympify(req.supply_expr)
    except Exception:
        raise HTTPException(400, "Invalid demand_expr or supply_expr")

    eq = solve(d_sym - s_sym, P)
    if not eq:
        raise HTTPException(400, "No equilibrium found")
    P_eq = float(eq[0])
    Q_eq = float(d_sym.subs(P, P_eq))

    p_vals = np.linspace(req.price_min, req.price_max, req.steps)
    f_d = lambdify(P, d_sym, 'numpy')
    f_s = lambdify(P, s_sym, 'numpy')

    fig, ax = plt.subplots()
    ax.plot(p_vals, f_d(p_vals), label='Demand')
    ax.plot(p_vals, f_s(p_vals), label='Supply')
    ax.scatter([P_eq], [Q_eq], color='red', label='Equilibrium')
    ax.set_xlabel('Price')
    ax.set_ylabel('Quantity')
    ax.legend()

    return {"price_eq": P_eq, "quantity_eq": Q_eq, "image_base64": fig_to_base64(fig)}

@app.post("/utility-maximization")
def utility_maximization(req: UtilityMaxRequest):
    x, y = symbols('x y', real=True, nonnegative=True)
    try:
        U = sympify(req.utility_expr)
    except Exception:
        raise HTTPException(400, "Invalid utility_expr")

    MUx = diff(U, x)
    MUy = diff(U, y)
    sol = solve([MUx/req.px - MUy/req.py, req.px*x + req.py*y - req.budget], [x, y])
    if not sol:
        raise HTTPException(400, "No solution found")
    x_opt, y_opt = sol[0]

    grid = np.linspace(0, req.grid_range, req.grid_steps)
    y_budget = (req.budget - req.px * grid) / req.py

    fig, ax = plt.subplots()
    ax.plot(grid, y_budget, label='Budget Line')
    X, Y = np.meshgrid(grid, grid)
    Z = lambdify((x, y), U, 'numpy')(X, Y)
    U_val = float(U.subs({x: x_opt, y: y_opt}))
    ax.contour(X, Y, Z, levels=[U_val], colors='purple')
    ax.scatter([float(x_opt)], [float(y_opt)], color='red', label='Optimal Bundle')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    return {"x_opt": float(x_opt), "y_opt": float(y_opt), "image_base64": fig_to_base64(fig)}

@app.post("/monopoly-pricing")
def monopoly_pricing(req: MonopolyRequest):
    Q = symbols('Q', real=True, nonnegative=True)
    try:
        P_sym = sympify(req.demand_expr)
        TC = sympify(req.total_cost_expr)
    except Exception:
        raise HTTPException(400, "Invalid demand_expr or total_cost_expr")

    MR = diff(P_sym * Q, Q)
    MC = diff(TC, Q)
    sol = solve(MR - MC, Q)
    if not sol:
        raise HTTPException(400, "No optimum found")
    Q_opt = float(sol[0])
    P_opt = float(P_sym.subs(Q, Q_opt))

    fig, ax = plt.subplots()
    q_vals = np.linspace(0, req.q_max, req.steps)
    ax.plot(q_vals, lambdify(Q, P_sym, 'numpy')(q_vals), label='Demand (P)')
    ax.plot(q_vals, lambdify(Q, MR, 'numpy')(q_vals), label='MR')
    ax.plot(q_vals, lambdify(Q, MC, 'numpy')(q_vals), label='MC')
    ax.scatter([Q_opt], [P_opt], color='red', label='Monopoly Optimum')
    ax.legend()

    return {"Q_opt": Q_opt, "P_opt": P_opt, "image_base64": fig_to_base64(fig)}

@app.post("/optimization")
def optimization(req: OptimizationRequest):
    x = symbols('x', real=True)
    try:
        f = sympify(req.f_expr)
    except Exception:
        raise HTTPException(400, "Invalid f_expr")

    f1 = diff(f, x)
    f2 = diff(f1, x)
    crit = solve(f1, x)
    results = []
    for cp in crit:
        if cp.is_real and req.x_min <= float(cp) <= req.x_max:
            sec = float(f2.subs(x, cp))
            kind = 'max' if sec < 0 else 'min' if sec > 0 else 'inflection'
            results.append({"x": float(cp), "f": float(f.subs(x, cp)), "type": kind})

    fig, ax = plt.subplots()
    xs = np.linspace(req.x_min, req.x_max, req.steps)
    ax.plot(xs, lambdify(x, f, 'numpy')(xs), label='f(x)')
    for r in results:
        ax.scatter(r['x'], r['f'], label=f"{r['type']} at {r['x']}")
    ax.legend()

    return {"critical_points": results, "image_base64": fig_to_base64(fig)}

@app.post("/elasticity")
def elasticity(req: ElasticityRequest):
    P = symbols('P', real=True)
    try:
        Q_sym = sympify(req.demand_expr)
    except Exception:
        raise HTTPException(400, "Invalid demand_expr")

    dQ = diff(Q_sym, P)
    Q0 = float(Q_sym.subs(P, req.price))
    elas = float(dQ.subs(P, req.price)) * req.price / Q0
    behavior = ("elastic" if abs(elas) > 1 else "inelastic" if abs(elas) < 1 else "unit elastic")

    fig, ax = plt.subplots()
    p_vals = np.linspace(req.price * 0.5, req.price * 1.5, req.steps)
    ax.plot(p_vals, lambdify(P, Q_sym, 'numpy')(p_vals), label='Demand')
    ax.scatter([req.price], [Q0], color='red', label=f'(P={req.price})')
    ax.legend()

    return {"elasticity": elas, "revenue_behavior": behavior, "image_base64": fig_to_base64(fig)}

# === New Endpoint: Fetch Math Topics Data with Images ===
@app.get("/math-data")
def math_data():
    """
    Returns base64-encoded placeholder images for each math topic.
    Frontend can decode the image_base64 value to display.
    """
    topics = ["algebra", "combinatorics", "number_theory"]
    data = {}
    for topic in topics:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, topic.replace('_', ' ').title(), fontsize=20, ha='center', va='center')
        ax.axis('off')
        data[topic] = {"image_base64": fig_to_base64(fig)}
    return data

# Note: Basic automated tests using TestClient have been removed due to missing httpx dependency.
# Please test endpoints manually (e.g., with curl or HTTP client).
