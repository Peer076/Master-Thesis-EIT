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
from pyeit.eit.fem import EITForward
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from IPython.display import Image, display

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
