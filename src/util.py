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

import matplotlib.pyplot as plt
import numpy as np

import imageio
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit import mesh
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from IPython.display import Image, display
from pyeit.eit.fem import EITForward, Forward
from pyeit.eit.interp2d import pdegrad, sim2pts
from PIL import Image
import os

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

    # Annotate each element with its index
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

def plot_3D_traj(sphere_r, tank_r, tank_h):

    r_path = 0.75
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_tank(tank_r, tank_h, ax)

    traj = "helix"
     
    N_steps = 100
    match traj:
        case "helix":
            helix_turns = 1
            theta = np.linspace(0, 2 * np.pi * helix_turns, N_steps)
            sphere_x = tank_r*r_path * np.cos(theta)
            sphere_y = tank_r*r_path * np.sin(theta)
            sphere_z = np.linspace(0, tank_h, N_steps)

        case "ellipse":
            tilt_angle = 45
            angle_rad = np.radians(tilt_angle)
            theta = np.linspace(0, 2*np.pi, N_steps)
            sphere_x = tank_r * np.cos(theta)
            sphere_y = (tank_r * np.sin(theta)) / np.cos(angle_rad)
            sphere_z = tank_h/2 * np.ones_like(theta)

    positions = np.column_stack((sphere_x, sphere_y, sphere_z))
    
    ax.plot(sphere_x, sphere_y, sphere_z, color='r')
    ax.scatter(sphere_x, sphere_y, sphere_z, color='b')
    
    ax.set_xlabel('x pos [mm]')
    ax.set_ylabel('y pos [mm]')
    ax.set_zlabel('z pos [mm]')
    ax.set_title('Kugel 3D Trajektorie')
     
    plt.show()
    
    return positions

def createTrajectory(traj,a,r_path, r_path_variations):
    # Erzeugt verschiedene Trajektorien-Pfade basierend auf dem 'traj'-Parameter

    if r_path_variations == True:

        bound = 0.2
        lower_bound = r_path * (1 - bound)  
        upper_bound = r_path * (1 + bound) 
        r_path = np.random.uniform(lower_bound, upper_bound)
        
    match traj:
            case "circle":
                    center=[np.cos(a)*r_path,np.sin(a)*r_path]
            case "flower":
                    n=2
                    r = np.sqrt(abs(np.cos(n*a)))
                    center=[np.cos(a)*r*r_path,np.sin(a)*r_path*r]
            case "eight":
                    center=[np.sin(a)*r_path,np.sin(2*a)*r_path/2]
            case "spiral":
                    spiral_r = r_path * (a / (2 * np.pi))  # Lineares Wachstum mit dem Winkel
                    center = [spiral_r * np.cos(a), spiral_r * np.sin(a)]
            case "pendel":
                    modified_angle = np.pi/np.sin(a)
                    center = [np.cos(modified_angle) * r_path, np.sin(modified_angle) * r_path]
            case "ellipse":
                    Rotation = True
                    b = r_path
                    c = r_path / 2
                    center = [b*np.cos(a),c*np.sin(a)]
                    if Rotation == True:
                          x = center[0]
                          center[0] = center[1]
                          center[1] = x
                    
    return center

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

def load_all_data(data_set):
    voltage_dict = {} 
    gamma_dict = {} 
    anomaly_dict = {}
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

        gamma_array = np.expand_dims(gamma_array, axis=2)
        
        
        voltage_dict[f"voltage{i}" if i > 0 else "voltage"] = voltage_array
        gamma_dict[f"gamma{i}" if i > 0 else "gamma"] = gamma_array
        anomaly_dict[f"anomaly{i}" if i > 0 else "anomaly"] = anomaly_array
        
    return voltage_dict, gamma_dict, anomaly_dict   

