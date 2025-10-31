"""Functions for Hw06 Q4
"""
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def generate_X():
    return np.random.randn(1000, 2)

def generate_Y(X):
    return 3 * X[:, 0] - 2 * X[:, 1] + np.random.uniform(-1, 1, size=(1000,))

def convex_objective(theta, X, y):
    return np.mean(np.abs(y - X @ theta))

def convex_grad(theta, X, y):
    return - (1/len(y)) * (X.T @ np.sign(y - X @ theta))

def nonconvex_objective(theta, X, y):
    c = np.array([-6.0, 6.0])
    mae = np.mean(np.abs(y - X @ theta))
    bump1 = -5.0 * np.exp(-0.25 * np.sum((theta - c)**2))
    total = mae + bump1
    return total

def nonconvex_grad(theta, X, y):
    n = len(y)
    c = np.array([-6.0, 6.0])
    df1_dtheta = - (1/n) * (X.T @ np.sign(y - X @ theta))
    df2_dtheta = 2.5 * (theta - c) * np.exp(-0.25 * np.sum((theta - c)**2))
    return df1_dtheta + df2_dtheta

def gradient_descent(obj_func_grad, X, y, initial_thetas, alpha, num_updates, batch_size = None):
    """Gradient Descent Algorithm"""
    n = len(X)
    if batch_size == None:
        batch_size = n
    thetas = np.array(initial_thetas).copy().astype(np.float64)
    path = [thetas.copy()]
    
    updates_done = 0
    while updates_done < num_updates:
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            if updates_done >= num_updates:
                break
            batch_idx = indices[start:start+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            grad = obj_func_grad(thetas, X_batch, y_batch)
            thetas -= alpha * grad
            path.append(thetas.copy())
            updates_done += 1
    return np.array(path)

def plot_surface_and_path(objective_func, X, y, path, title):
    x_max = path[:,0].max()
    x_min = path[:,0].min()
    y_max = path[:,1].max()
    y_min = path[:,1].min()
    if (x_max >= 10) or (x_min <= -10):
        x_top = x_max + 2 if x_max >= 10 else 10
        x_bottom = x_min - 2 if x_min <= -10 else -10
    else:
        x_top = 10
        x_bottom = -10
    if (y_max >= 10) or (y_min <= -10):
        y_top = y_max + 2 if y_max >= 10 else 10
        y_bottom = y_min - 2 if y_min <= -10 else -10
    else:
        y_top = 10
        y_bottom = -10 
    x_vals = np.linspace(x_bottom, x_top, 100)
    y_vals = np.linspace(y_bottom, y_top, 100)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    Z_grid = np.zeros_like(X_grid)

    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            Z_grid[i, j] = objective_func(np.array([X_grid[i, j], Y_grid[i, j]]), X, y)

    # Create 3D surface plot
    surface = go.Surface(x=X_grid, y=Y_grid, z=Z_grid, opacity=0.8, colorscale="Viridis", showscale=False)

    path_trace = go.Scatter3d(
        x=path[:, 0],
        y=path[:, 1],
        z=[objective_func(p, X, y) for p in path],
        mode="lines+markers",
        line=dict(color="red", width=3),
        marker=dict(size=5, color="red"),
        name="Path"
    )
    start_point_3D = go.Scatter3d(
    x=[path[0, 0]],
    y=[path[0, 1]],
    z=[objective_func(path[0], X, y)],
    mode="markers",
    marker=dict(size=5, color="blue", symbol="circle"),
    name="Start Point"
    )
    

    # Create contour plot
    contour = go.Contour(
        x=x_vals,
        y=y_vals,
        z=Z_grid,
        colorscale="Viridis",
        showscale=False,
        contours=dict(showlabels=True),
        line_smoothing=0.85
    )
    path_2d = go.Scatter(
        x=path[:, 0],
        y=path[:, 1],
        mode="lines+markers",
        line=dict(color="red", width=2),
        marker=dict(size=8, color="red"),
        name="Path"
    )
    start_point_2D = go.Scatter(
    x=[path[0, 0]],
    y=[path[0, 1]],
    mode="markers",
    marker=dict(size=8, color="blue", symbol="circle"),
    name="Start Point"
    )

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("3D Surface", "Contour Plot"),
                        specs=[[{'type': 'surface'}, {'type': 'xy'}]])

    fig.add_trace(surface, row=1, col=1)
    fig.add_trace(path_trace, row=1, col=1)
    fig.add_trace(start_point_3D, row=1, col=1)
    fig.add_trace(contour, row=1, col=2)
    fig.add_trace(path_2d, row=1, col=2)
    fig.add_trace(start_point_2D, row=1, col=2)

    fig.update_layout(title_text=title, height=600, width=1000)
    fig.show()

def plot_function(title, X, y, convex = True):

    x_vals = np.linspace(-10, 10, 100)
    y_vals = np.linspace(-10, 10, 100)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    Z_grid = np.zeros_like(X_grid)

    if convex:
        objective_func = convex_objective
    else:
        objective_func = nonconvex_objective

    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            Z_grid[i, j] = objective_func(np.array([X_grid[i, j], Y_grid[i, j]]), X, y)

    # Create 3D surface plot
    surface = go.Surface(x=X_grid, y=Y_grid, z=Z_grid, opacity=0.8, colorscale="Viridis", showscale=False)

    # Create contour plot
    contour = go.Contour(
        x=x_vals,
        y=y_vals,
        z=Z_grid,
        colorscale="Viridis",
        showscale=False,
        contours=dict(showlabels=True),
        line_smoothing=0.85
    )

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("3D Surface", "Contour Plot"),
                        specs=[[{'type': 'surface'}, {'type': 'xy'}]])

    fig.add_trace(surface, row=1, col=1)
    fig.add_trace(contour, row=1, col=2)

    fig.update_layout(title_text=title, height=600, width=1000)
    fig.show()


def visualize_gradient_descent(title, X, y, initial_theta, alpha, num_updates, batch_size = None, convex = True):
    if convex:
        path = gradient_descent(convex_grad, X, y, initial_theta, alpha, num_updates, batch_size)
        plot_surface_and_path(convex_objective, X, y, path, title)
    else: 
        path = gradient_descent(nonconvex_grad, X, y, initial_theta, alpha, num_updates, batch_size)
        plot_surface_and_path(nonconvex_objective, X, y, path, title)