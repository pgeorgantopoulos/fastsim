import numpy as np
import h5py
import dash
from dash import dcc, html
import plotly.graph_objects as go

# Load a sample event
h5_path = "/home/panos/calo-data/dataset_1_photons_1.hdf5"
event_idx = 0

with h5py.File(h5_path, 'r') as f:
    showers = f['showers'][:]
event = showers[event_idx]  # shape (num_z, num_alpha, num_r)
print(event.shape)
volume = np.transpose(event, (2, 1, 0))  # (r, alpha, z)

nx, ny, nz = volume.shape
x, y, z = np.indices((nx, ny, nz))

# Build Plotly figure
fig = go.Figure(data=go.Volume(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=volume.flatten(),
    opacity=0.1,
    surface_count=25,
    colorscale='Viridis'
))

fig.update_layout(scene=dict(
    xaxis_title='Radial',
    yaxis_title='Angular',
    zaxis_title='Layer'
))

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("CaloChallenge 3D Shower Viewer"),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
