import json
import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Tuple, Union
import imageio
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit import mesh
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from IPython.display import Image, display
from pyeit.eit.fem import EITForward, Forward
from pyeit.eit.interp2d import pdegrad, sim2pts
from PIL import Image
from scipy.integrate import cumulative_trapezoid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA


def define_mesh_obj(n_el,use_customize_shape):
    n_el = 16  # nb of electrodes
    use_customize_shape = False
    if use_customize_shape:
        # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax
        mesh_obj = mesh.create(n_el, h0=0.05, fd=thorax)
    else:
        mesh_obj = mesh.create(n_el, h0=0.05)
                   # Elektrodenpositionen extrahieren

# Extrahiert Informationen über das Maschennetz wie Elektrodenpositionen, Knoten und Elemente
    return mesh_obj

##########################
def plot_mesh(
    mesh_obj, figsize: tuple = (6, 4), title: str = "mesh"
) -> None:
    """
    Plot a given PyEITMesh mesh object.

    Parameters
    ----------
    mesh_obj : PyEITMesh
        mesh object
    figsize : tuple, optional
        figsize, by default (6, 4)
    title : str, optional
        title of the figure, by default "mesh"
    """
    plt.style.use("default")
    pts = mesh_obj.node
    tri = mesh_obj.element
    x, y = pts[:, 0], pts[:, 1]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.tripcolor(
        x,
        y,
        tri,
        np.real(mesh_obj.perm_array),
        edgecolors="k",
        shading="flat",
        alpha=0.5,
        cmap=plt.cm.viridis,
    )
    # draw electrodes
    ax.plot(x[mesh_obj.el_pos], y[mesh_obj.el_pos], "ro")
    for i, e in enumerate(mesh_obj.el_pos):
        ax.text(x[e], y[e], str(i + 1), size=12)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])
    fig.set_size_inches(6, 6)
    plt.show()

####################################
def load_data(data_set: int, mat_complex=False, info=True):
    timestamp = list()
    perm_array = list()
    d = list()
    eit = list()
    for ele in np.sort(glob(f"../data/#{data_set}/DATA/*.npz")):
        tmp = np.load(ele, allow_pickle=True)
        timestamp.append(tmp["timestamp"])
        perm_array.append(tmp["perm_arr"])
        d.append(tmp["d"])
        eit.append(tmp["eit"])
    timestamp = np.array(timestamp)
    perm_array = np.array(perm_array)
    d = np.array(d)
    eit = np.array(eit)
    eit = z_score_normalization(eit)
    if not mat_complex:
        eit = np.abs(eit)
    if info:
        # Load and display the image
        print(
            f"Time of\n\tfirst: {convert_timestamp(min(timestamp))}\n\tlast: {convert_timestamp(max(timestamp))}\nmeasurement."
        )
        print(f"Shape of EIT data: {eit.shape}")
        print(f"Shape of Permittivity Data: {perm_array.shape}")
        img = Image(filename=f"../data/#{data_set}/PCA.png")
        display(img)
    return eit, perm_array, d, timestamp

def pca(V,angle):

    pca = PCA(n_components=2)
    V_pca = pca.fit_transform(V)

    plt.figure(figsize=(10,6))
    scatter = plt.scatter(V_pca[:, 0], V_pca[:, 1], c=angle, cmap='viridis')
    plt.colorbar(scatter,label='Winkel (Grad)')
    plt.xlabel('Hauptkomponente 1 (PC1)')
    plt.ylabel('Hauptkomponente 2 (PC2)')
    plt.show()

def z_score_normalization(data, axis=(1, 2)):
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    return (data - mean) / std


def convert_timestamp(date_str):
    if len(str(date_str).split(".")) > 2:
        timestamp = datetime.strptime(date_str, "%Y.%m.%d. %H:%M:%S.%f")
        return timestamp.timestamp()
    else:
        date_time = datetime.fromtimestamp(float(date_str))
        return date_time.strftime("%Y.%m.%d. %H:%M:%S.%f")


def get_fps(timestamps):
    diff = np.diff(timestamps)
    fps = 1 / diff
    print(f"Mean fps of {np.mean(fps):.2f}")


# Function to ensure that a particular directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_empty_mesh(
    mesh_obj, perm_array, ax=None, title="Empty Mesh", sample_index=None
):
    el_pos = np.arange(mesh_obj.n_el)
    pts = mesh_obj.node
    tri = mesh_obj.element
    x, y = pts[:, 0], pts[:, 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    vmin, vmax = 0, 10

    im = ax.tripcolor(
        x,
        y,
        tri,
        perm_array,
        shading="flat",
        edgecolor="k",
        alpha=0.8,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    # Annotate each element with its index
    for j, el in enumerate(el_pos):
        ax.text(pts[el, 0], pts[el, 1], str(j + 1), color="red", fontsize=8)

    ax.set_aspect("equal")
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])

    if sample_index is not None:
        ax.set_title(f"{title} Sample {sample_index}")
    else:
        ax.set_title(title)

    # Create colorbar with limits
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Permittivity")
    cbar.ax.tick_params(labelsize=8)

    if ax is None:
        plt.show()


# Function to plot mesh
def plot_mesh_permarray(mesh_obj, perm_array, ax=None, title="Mesh", sample_index=None):
    el_pos = np.arange(mesh_obj.n_el)
    pts = mesh_obj.node
    tri = mesh_obj.element
    x, y = pts[:, 0], pts[:, 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.tripcolor(x, y, tri, perm_array, shading="flat", edgecolor="k", alpha=0.8)

    for j, el in enumerate(el_pos):
        ax.text(pts[el, 0], pts[el, 1], str(j + 1), color="red")

    ax.set_aspect("equal")
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])

    if sample_index is not None:
        ax.set_title(f"{title} Sample {sample_index}")
    else:
        ax.set_title(title)
    if ax is None:
        fig.colorbar(im, ax=ax)
        plt.show()
    else:
        plt.colorbar(im, ax=ax)


def seq_data(eit, perm, n_seg=4):
    sequence = [eit[i : i + n_seg] for i in range(len(eit) - n_seg)]
    aligned_perm = perm[n_seg:]
    return np.array(sequence), np.array(aligned_perm)


def plot_tank(r, h, ax):
    
    theta = np.linspace(0, 2 * np.pi, 100)
    z_cylinder = np.linspace(0, h, 100)
    X_cylinder, Z_cylinder = np.meshgrid(r * np.cos(theta), z_cylinder)
    Y_cylinder, _ = np.meshgrid(r * np.sin(theta), z_cylinder)
    ax.plot_surface(X_cylinder, Y_cylinder, Z_cylinder, color='lightgray', alpha=0.5)


def interpolate_equidistant_points(x, y, num_points):
    """
    Interpolate points along the curve to create equidistant spacing
    """
    dx = np.diff(x)
    dy = np.diff(y)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    
    target_lengths = np.linspace(0, cumulative_lengths[-1], num_points)
    
    interp_x = interp1d(cumulative_lengths, x, kind='linear')
    interp_y = interp1d(cumulative_lengths, y, kind='linear')
    
    return interp_x(target_lengths), interp_y(target_lengths)

def create_trajectory(traj_type, radius, num_points, base_rotations=1):
    """
    Create a trajectory with constant point spacing based on reference circle
    
    Parameters:
    traj_type (str): Type of trajectory ('Kreis', 'Ellipse', 'Acht', 'Spirale', 'Schlange')
    radius (float): Base radius/size of the trajectory
    base_points (int): Number of points for reference circle
    reference_radius (float): Radius of reference circle (default 0.25)
    base_rotations (int): Number of rotations for spiral (default 2)
    
    Returns:
    numpy.ndarray: Array of points with consistent spacing
    """
    
    t = np.linspace(0, 2*np.pi, num_points)
    
    if traj_type == "Kreis":
        x_uniform = radius * np.cos(t)
        y_uniform = radius * np.sin(t)
        
    elif traj_type == "Ellipse":
        a = radius  
        b = 0.7 * radius 
        x = a * np.cos(t)
        y = b * np.sin(t)
        
    elif traj_type == "Acht":
        x = radius * np.sin(t)
        y = radius * np.sin(2*t) / 2
        x_uniform, y_uniform = interpolate_equidistant_points(x, y, num_points)
    
    elif traj_type == "Spirale":
        rotations = base_rotations + (num_points // 1000)
        t = np.linspace(0, 2*np.pi*rotations, 1000)
        
        r = radius * (1 - t/(2*np.pi*rotations))
        x = r * np.cos(t)
        y = r * np.sin(t)
        
        x_uniform, y_uniform = interpolate_equidistant_points(x, y, num_points)

    elif traj_type == "Lissajous":
        a = 3  # Frequenz x
        b = 4  # Frequenz y
        t = np.linspace(0, 2*np.pi, 1000)
        
        x = radius * np.sin(a * t)
        y = radius * np.sin(b * t)
        
        x_uniform, y_uniform = interpolate_equidistant_points(x, y, num_points)

    elif traj_type == "Folium_vorwärts":
        circle_circumference = 2 * np.pi * radius
        
        t = np.linspace(-3, 3, 1000)  
        
        a = radius * 0.5  
        
        x = a * (t**3 - 3*t) / (1 + t**2)
        y = a * (t**2 - 1) / (1 + t**2)
        
        dx = np.diff(x)
        dy = np.diff(y)
        current_length = np.sum(np.sqrt(dx**2 + dy**2))
        
        scale = np.sqrt(circle_circumference / current_length)
        x *= scale
        y *= scale
        
        max_radius = np.max(np.sqrt(x**2 + y**2))
        if max_radius > radius:
            scaling_factor = (radius / max_radius) * 0.95  
            x *= scaling_factor
            y *= scaling_factor
        
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        
        x_uniform, y_uniform = interpolate_equidistant_points(x, y, num_points)

    elif traj_type == "Folium_rückwärts":
        circle_circumference = 2 * np.pi * radius
        
        t = np.linspace(-3, 3, 1000)  
        
        a = radius * 0.5  
        
        x = a * (t**3 - 3*t) / (1 + t**2)
        y = a * (t**2 - 1) / (1 + t**2)
    
        dx = np.diff(x)
        dy = np.diff(y)
        current_length = np.sum(np.sqrt(dx**2 + dy**2))
        
        scale = np.sqrt(circle_circumference / current_length)
        x *= scale
        y *= scale

        max_radius = np.max(np.sqrt(x**2 + y**2))
        if max_radius > radius:
            scaling_factor = (radius / max_radius) * 0.95  
            x *= scaling_factor
            y *= scaling_factor
        
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        x = x[::-1]
        y = y[::-1]
        
        x_uniform, y_uniform = interpolate_equidistant_points(x, y, num_points)

    
    return np.column_stack((x_uniform, y_uniform))


def create_trajectory_3D(traj_type, radius, num_points, base_rotations=2):

    if traj_type == "ModulatedHelix":
        t = np.linspace(0, 2*np.pi*base_rotations, num_points)
    
        osc_amplitude = 97*0.1
        osc_frequency = 5  
        phase_shift = 0
    
        # Die Grundform der Helix bleibt gleich
        R = radius + osc_amplitude * np.sin(t * osc_frequency)  # Vereinfachte Oszillation
        x = R * np.cos(t)
        y = R * np.sin(t)
        z = np.linspace(50, 100, num_points)

    elif traj_type == "Helix":
        t = np.linspace(0, 2*np.pi*base_rotations, num_points)
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = np.linspace(50, 100, num_points)

    elif traj_type == "Lorenz":
        
        sigma = 10
        rho = 28
        beta = 8/3
        

        dt = 0.01
        t = np.linspace(0, dt*num_points, num_points)
        
        x = np.zeros(num_points)
        y = np.zeros(num_points)
        z = np.zeros(num_points)
        
        x[0], y[0], z[0] = 1, 1, 1
        
        for i in range(num_points-1):
            dx = sigma * (y[i] - x[i])
            dy = x[i] * (rho - z[i]) - y[i]
            dz = x[i] * y[i] - beta * z[i]
            
            x[i+1] = x[i] + dx * dt
            y[i+1] = y[i] + dy * dt
            z[i+1] = z[i] + dz * dt
        
        x = (x - x.min()) / (x.max() - x.min()) * radius * 2 - radius
        y = (y - y.min()) / (y.max() - y.min()) * radius * 2 - radius
        z = (z - z.min()) / (z.max() - z.min()) * 50 + 50  # Scale to range [50, 100]

    elif traj_type == "Spherical":
        # Parameter für die sphärische Spirale
        radius_sphere = radius  # Radius der Kugel
        n_turns = 60  # Anzahl der Windungen von oben nach unten
        
        # Parameter equations für eine sphärische Spirale
        theta = np.linspace(0, 2*np.pi*n_turns, num_points)  # Azimuthalwinkel
        phi = np.linspace(0, np.pi, num_points)  # Polarwinkel
        
        # Berechnung der kartesischen Koordinaten
        x = radius_sphere * np.sin(phi) * np.cos(theta)
        y = radius_sphere * np.sin(phi) * np.sin(theta)
        z = radius_sphere * np.cos(phi)
        
        # Skalierung und Verschiebung des z-Wertes auf den gewünschten Bereich [50, 100]
        z = (z - z.min()) / (z.max() - z.min()) * 50 + 50
        
    
    return np.column_stack((x, y, z))
        
###
def create2DAnimation(traj,mesh_new_list, protocol_obj,mesh_obj,output_gif="animation_with_movement.gif"):
    pts = mesh_obj.node                         # Knoten extrahieren
    tri = mesh_obj.element                      # Elemente extrahieren
    x, y = pts[:, 0], pts[:, 1]                 # x- und y-Koordinaten trennen
    image_files = []
    output_gif = f"animation_{traj}.gif"
    # Loop durch jede Anomalie-Position
    center_coords = []
    for i, mesh_data in enumerate(mesh_new_list):
        center_coords.append([mesh_data["x"], mesh_data["y"]])
        # Loop durch jedes Elektrodenpaar für die aktuelle Anomalie-Position
        for j, ex_line in enumerate(protocol_obj.ex_mat):
            fig, ax1 = plt.subplots(figsize=(9, 6))

            # Potentialverteilung für dieses Elektrodenpaar
            fwd = Forward(mesh_data["mesh"])
            f = np.real(fwd.solve(ex_line.ravel()))

            # Äquipotentiallinien basierend auf der Einspeisung
            vf = np.linspace(min(f), max(f), 64)
            ax1.tricontour(x, y, tri, f, vf, cmap=plt.cm.viridis)

            ax1.tripcolor(
                x,
                y,
                tri,
                np.real(mesh_data["mesh"].perm),
                edgecolors="k",
                shading="flat",
                alpha=0.5,
                cmap=plt.cm.Greys,
            )

            # plottet Elektroden
            ax1.plot(x[mesh_obj.el_pos], y[mesh_obj.el_pos], "ro")
            for e_idx, e in enumerate(mesh_obj.el_pos):
                ax1.text(x[e], y[e], str(e_idx + 1), size=12)
                
            center_x = [coord[0] for coord in center_coords]
            center_y = [coord[1] for coord in center_coords]
            ax1.plot(center_x, center_y, marker='o', color='b', label="Trajectory Path")

            # Plotte das Zentrum der aktuellen Anomalie
            ax1.scatter(mesh_data["x"], mesh_data["y"], color="r", label="Anomaly Center")
            ax1.set_title(f"Trajectory-Path:{ traj}, Injection Electrode-Pair: [{ex_line[0]};{ex_line[1]}]")
            ax1.set_aspect("equal")
            ax1.set_ylim([-1.2, 1.2])
            ax1.set_xlim([-1.2, 1.2])
            fig.set_size_inches(6, 6)

            # Speichert das Bild
            filename = f"frame_{i}_{j}.png"
            plt.savefig(filename)
            image_files.append(filename)
            plt.close(fig)

    # Erstellt das GIF
    frames = [Image.open(image) for image in image_files]
    frames[0].save(output_gif, format="GIF", append_images=frames[1:], save_all=True, duration=100, loop=0)

    # Einzelbilder löschen
    for image in image_files:
        os.remove(image)
###
def load_sim_data(data_set):
    data_dirs = sorted(glob(f"data_set/{data_set}/"))  

    for i, directory in enumerate(data_dirs):
        file_list = sorted(glob(f"{directory}*.npz"))  
        voltage_list = []
        gamma_list = []
        anomaly_list = []

        for file in file_list:
            tmp = np.load(file, allow_pickle=True)  
            voltage_list.append(tmp["v"])  
            gamma_list.append(tmp["gamma"])
            anomaly_list.append(tmp["anomaly"])

        voltage_array = np.array(voltage_list) 
        anomaly_array = np.array(anomaly_list)
        gamma_array = np.array(gamma_list)
    
        
    return voltage_array, gamma_array, anomaly_array  

def load_exp_data(data_set):
    data_dirs = sorted(glob(f"exp_data_set/{data_set}/"))  

    for i, directory in enumerate(data_dirs):
        file_list = sorted(glob(f"{directory}*.npz"))  
        voltage_list = []
        temp_list = []
        timestamp_list = []
        position_list = []
     
        for file in file_list:
            tmp = np.load(file, allow_pickle=True)  
            voltage_list.append(tmp["v"])
            temp_list.append(tmp["temperature"])
            timestamp_list.append(tmp["timestamp"])
            position_list.append(tmp["position"])
            

        voltage_array = np.array(voltage_list)
        temp_array = np.array(temp_list)
        timestamp_array = np.array(timestamp_list)
        position_array = np.array(position_list)
        
    return voltage_array, temp_array, timestamp_array, position_array

def load_exp_data3D(data_set):
    data_dirs = sorted(glob(f"exp_data_set_3D/{data_set}/"))  

    for i, directory in enumerate(data_dirs):
        file_list = sorted(glob(f"{directory}*.npz"))  
        voltage_list = []
        temp_list = []
        timestamp_list = []
        position_list = []
     
        for file in file_list:
            tmp = np.load(file, allow_pickle=True)  
            voltage_list.append(tmp["v"])
            temp_list.append(tmp["temperature"])
            timestamp_list.append(tmp["timestamp"])
            position_list.append(tmp["position"])
            

        voltage_array = np.array(voltage_list)
        temp_array = np.array(temp_list)
        timestamp_array = np.array(timestamp_list)
        position_array = np.array(position_list)
        
    return voltage_array, temp_array, timestamp_array, position_array
###
#Function to create mesh plots and save them for comparison
def mesh_plot_comparisons(
    mesh_obj,
    selected_indices,
    selected_true_perms,
    selected_predicted_perms,
    save_dir="comparison_plots",
    gif_name="comparison.gif",
    gif_title="Mesh Comparison",
    fps=1,
):
    os.makedirs(save_dir, exist_ok=True)

    images = []

    for i in range(len(selected_indices)):
        true_perm = selected_true_perms[i].flatten()
        pred_perm = selected_predicted_perms[i].flatten()

        assert len(true_perm) == len(
            mesh_obj.element
        ), f"Length of true_perm ({len(true_perm)}) does not match number of elements ({len(mesh_obj.element)})"
        assert len(pred_perm) == len(
            mesh_obj.element
        ), f"Length of pred_perm ({len(pred_perm)}) does not match number of elements ({len(mesh_obj.element)})"

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(gif_title, fontsize=16)

        plot_mesh_permarray(
            mesh_obj,
            true_perm,
            ax=axs[0],
            title="Original",
            sample_index=selected_indices[i],
        )
        plot_mesh_permarray(
            mesh_obj,
            pred_perm,
            ax=axs[1],
            title="Predicted",
            sample_index=selected_indices[i],
        )

        filename = os.path.join(save_dir, f"comparison_{i + 1}.png")
        plt.savefig(filename, format="png", dpi=300)
        plt.savefig(filename + ".pdf")
        plt.show()

        images.append(imageio.imread(filename))
        plt.close(fig)

    gif_path = os.path.join(save_dir, gif_name)
    duration_per_frame = 1000 / fps
    imageio.mimsave(gif_path, images, duration=duration_per_frame, loop=0)

    png_dats = glob(os.path.join(save_dir, "*.png"))
    for dat in png_dats:
        os.remove(dat)
###
def plot_boxplot(
    data, ylabel, title, savefig_name, save_dir="plots", figsize=(6, 8), dpi=300
):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=figsize)
    plt.boxplot(data)
    plt.ylabel(ylabel)
    plt.title(title)
    save_path = os.path.join(save_dir, savefig_name)
    plt.savefig(save_path, format="png", dpi=dpi)
    plt.show()
###
# Function to select a number of random instances for mesh plots comparison
def select_random_instances(x_test, y_test, predicted_permittivities, num_instances=10):
    random_indices = random.sample(range(x_test.shape[0]), num_instances)
    selected_true_perms = y_test[random_indices]
    selected_predicted_perms = predicted_permittivities[random_indices]
    return random_indices, selected_true_perms, selected_predicted_perms

###
def calculate_perm_error(X_true, X_pred):
    perm_error = list()
    obj_threshold = (np.max(X_true) - np.min(X_true)) / 2
    mesh_obj = mesh.create(n_el=32, h0=0.05)

    for perm_true, perm_pred in zip(X_true, X_pred):
        perm_error.append(
            compute_perm_deviation(
                mesh_obj, perm_true, perm_pred, obj_threshold, plot=False
            )
        )
    perm_error = np.array(perm_error)

    return perm_error

# Functions to compute FEM deviations for box plot
def compute_perm_deviation(
    mesh_obj,
    perm_true: np.ndarray,
    perm_pred: np.ndarray,
    obj_threshold: Union[int, float],
    plot: bool = False,
) -> int:
    # Identify object indices based on threshold
    obj_idx_true = np.where(perm_true > obj_threshold)[0]
    obj_idx_pred = np.where(perm_pred > obj_threshold)[0]

    perm_dev = len(obj_idx_pred) - len(obj_idx_true)

    return perm_dev

# Auswahl zufälliger Instanzen unter Verwendung der Indizes aus test_indices
def select_random_instances_mapper(x_test, y_test, predicted_permittivities, indices, num_instances=10):
    random_indices = random.sample(range(x_test.shape[0]), num_instances)
    selected_true_perms = y_test[random_indices]
    selected_predicted_perms = predicted_permittivities[random_indices]
    selected_indices = indices[random_indices]  # Hole die tatsächlichen Indizes
    return selected_indices, selected_true_perms, selected_predicted_perms

import numpy as np
import json
import random
import matplotlib.pyplot as plt
from .classes import (
    BallAnomaly,
    Boundary,
)


def voxel_ball(ball, boundary, empty_gnd=0, mask=False):
    y, x, z = np.indices((boundary.x_length, boundary.y_length, boundary.z_length))
    voxel = (
        np.sqrt((x - ball.x) ** 2 + (y - ball.y) ** 2 + (z - ball.z) ** 2) < ball.d / 2
    )
    if mask:
        return voxel
    else:
        return np.where(voxel, ball.perm, empty_gnd)


def plot_reconstruction_set(
    true, m_true, pred, m_pred, cols=5, legends=False, save_fig=None, forced_sel=None
):
    if true.shape != pred.shape:
        print("true.shape != pred.shape")
        return

    rows = 2
    colors = ["C0", "C1"]  # Define colors for 1 and 2 values respectively
    if forced_sel is None:
        sel = random.sample(range(true.shape[0]), cols)
    else:
        sel = forced_sel
    print("Selcted the samples =", sel)
    fig, axes = plt.subplots(
        rows, cols, figsize=(14, 5), subplot_kw={"projection": "3d"}
    )
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            voxelarray = (
                true[sel[j]] * (1 + m_true[sel[j]])
                if i == 0
                else pred[sel[j]] * (1 + m_pred[sel[j]])
            )
            ax.voxels(voxelarray, facecolors=colors[int(np.max(voxelarray) - 1)])
            if legends:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
            ax.view_init(azim=45, elev=30)
    if not legends:
        print("Row 0 -> true γ distribution")
        print("Row 1 -> pred γ distribution")
    # plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig, bbox_inches="tight", pad_inches=0)
    plt.show()


def read_json_file(file_path):
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def plot_voxel_c(voxelarray, elev=20, azim=10):
    """
    fc : facecolor of the voxels
    """
    # C0 -> acrylic
    # C1 -> metal
    colors = ["C0", "C1"]  # Define colors for 1 and 2 values respectively

    ax = plt.figure(figsize=(4, 4)).add_subplot(projection="3d")
    # ax.voxels(voxelarray.transpose(1, 0, 2))
    ax.voxels(
        voxelarray.transpose(1, 0, 2), facecolors=colors[int(np.max(voxelarray) - 1)]
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(azim=azim, elev=elev)
    plt.tight_layout()
    plt.show()


def voxel_ball(ball, boundary, empty_gnd=0, mask=False):
    
    #scale_factor = 32 / 194  # 32 voxels / 194mm (tank diameter)
    #ball_diameter_voxels = round(ball.d * scale_factor)
    #print(f"ball_diameter_voxels: {ball_diameter_voxels}")
    
    y, x, z = np.indices((boundary.x_length, boundary.y_length, boundary.z_length))
    voxel = (
        np.sqrt((x - ball.x) ** 2 + (y - ball.y) ** 2 + (z - ball.z) ** 2) < ball.d / 2
    )
    if mask:
        return voxel
    else:
        return np.where(voxel, ball.perm, empty_gnd)

def create_cylinder_mesh(tank, n_points=100):
    
    theta = np.linspace(0, 2*np.pi, n_points)
    z = np.linspace(tank.T_bz[0], tank.T_bz[1], n_points)
    theta, z = np.meshgrid(theta, z)
    x = tank.T_r * np.cos(theta)
    y = tank.T_r * np.sin(theta)
    return x, y, z

def plot_tank_and_ball(ball, tank, boundary):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
   
    scale_factor = 32 / tank.T_d 
    
    cylinder_radius = tank.T_r * scale_factor
    cylinder_height = tank.T_bz[1] * scale_factor
    x_cyl, y_cyl, z_cyl = create_cylinder_mesh(cylinder_radius, cylinder_height)
    
    x_cyl += 16  
    y_cyl += 16  
    
    ax.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.1, color='gray')
    
    ball_voxels = voxel_ball(ball, boundary)
    
    ax.voxels(ball_voxels.transpose(1, 0, 2), facecolors='cornflowerblue', alpha=0.8)
    
    ax.plot([0, boundary.x_length], [0, 0], [0, 0], 'k-', linewidth=1)
    #ax.text(boundary.x_length+1, 0, 0, 'x')
    
    ax.plot([0, 0], [0, boundary.y_length], [0, 0], 'k-', linewidth=1)
    #ax.text(0, boundary.y_length+1, 0, 'y')
    
    ax.plot([0, 0], [0, 0], [0, boundary.z_length], 'k-', linewidth=1)
    #ax.text(0, 0, boundary.z_length+1, 'z')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.set_xlim([boundary.x_0, boundary.x_length])
    ax.set_ylim([boundary.y_0, boundary.y_length])
    ax.set_zlim([boundary.z_0, boundary.z_length])
    
    # Set grid with minor lines
    ax.grid(True)
    
    # Set major ticks to show 0,8,16,24,32
    ax.set_xticks([0, 8, 16, 24, 32])
    ax.set_yticks([0, 8, 16, 24, 32])
    ax.set_zticks([0, 8, 16, 24, 32])
    
    # Set view angle
    ax.view_init(elev=10, azim=45)
    
    plt.tight_layout()
    plt.show()

def plot_ball(
    ball: BallAnomaly,
    boundary: Boundary,
    res: int = 50,
    elev: int = 25,
    azim: int = 10,
):
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(0, np.pi, res)

    x_c = ball.x + ball.d / 2 * np.outer(np.cos(u), np.sin(v))
    y_c = ball.y + ball.d / 2 * np.outer(np.sin(u), np.sin(v))
    z_c = ball.z + ball.d / 2 * np.outer(np.ones(np.size(u)), np.cos(v))

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    # ball
    ax.plot_surface(x_c, y_c, z_c, color="C0", alpha=1)

    ax.set_xlim([boundary.x_0, boundary.x_length])
    ax.set_ylim([boundary.y_0, boundary.y_length])
    ax.set_zlim([boundary.z_0, boundary.z_length])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()


def plot_voxel_c(voxelarray, elev=20, azim=10):
    """
    fc : facecolor of the voxels
    """
    # C0 -> acrylic
    # C1 -> metal
    colors = ["C0", "C1"]  # Define colors for 1 and 2 values respectively

    ax = plt.figure(figsize=(4, 4)).add_subplot(projection="3d")
    # ax.voxels(voxelarray.transpose(1, 0, 2))
    ax.voxels(
        voxelarray.transpose(1, 0, 2), facecolors=colors[int(np.max(voxelarray) - 1)]
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(azim=azim, elev=elev)
    plt.tight_layout()
    plt.show()


def plot_voxel(voxelarray, fc=0, elev=20, azim=10):
    ax = plt.figure(figsize=(4, 4)).add_subplot(projection="3d")
    ax.voxels(voxelarray.transpose(1, 0, 2), facecolors=f"C{fc}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(azim=azim, elev=elev)
    plt.tight_layout()
    plt.show()


def plot_reconstruction_set(
    true, m_true, pred, m_pred, cols=5, legends=False, save_fig=None, forced_sel=None
):
    if true.shape != pred.shape:
        print("true.shape != pred.shape")
        return

    rows = 2
    colors = ["C0", "C1"]  # Define colors for 1 and 2 values respectively
    if forced_sel is None:
        sel = random.sample(range(true.shape[0]), cols)
    else:
        sel = forced_sel
    print("Selcted the samples =", sel)
    fig, axes = plt.subplots(
        rows, cols, figsize=(14, 5), subplot_kw={"projection": "3d"}
    )
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            voxelarray = (
                true[sel[j]] * (1 + m_true[sel[j]])
                if i == 0
                else pred[sel[j]] * (1 + m_pred[sel[j]])
            )
            ax.voxels(voxelarray, facecolors=colors[int(np.max(voxelarray) - 1)])
            if legends:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
            ax.view_init(azim=45, elev=30)
    if not legends:
        print("Row 0 -> true γ distribution")
        print("Row 1 -> pred γ distribution")
    # plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig, bbox_inches="tight", pad_inches=0)
    plt.show()


def read_json_file(file_path):
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")