from tensorflow.keras.models import load_model
from glob import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from src.vae_model import vae_model
from src.lstm_mapper_model import mapper_model
from src.util import (
    seq_data, 
    load_sim_data,  
    compute_perm_deviation, 
    calculate_perm_error, 
    select_random_instances,
    plot_boxplot,
    mesh_plot_comparisons,
    plot_mesh_permarray,
    load_exp_data,
    plot_mesh
)
from pyeit import mesh
from keras import backend as K
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras.models import Model
from tqdm import tqdm
import pandas as pd
import seaborn as sns

def compute_evaluation_metrics(mesh_obj, true_perm, pred_perm, threshold=0.5):
    
    # Calculate triangle centers
    tri_centers = np.mean(mesh_obj.node[mesh_obj.element], axis=1)
    
    # Get indices where values exceed threshold
    true_indices = np.where(true_perm > threshold)[0]
    pred_indices = np.where(pred_perm > threshold)[0]
    
    # Check if either array is empty
    if len(true_indices) == 0 or len(pred_indices) == 0:
        return None, None
    
    # Get coordinates for elements above threshold
    true_coords = tri_centers[true_indices]
    pred_coords = tri_centers[pred_indices]
    
    # Calculate mean positions (centers)
    true_center = np.mean(true_coords, axis=0)
    pred_center = np.mean(pred_coords, axis=0)
    
    # Check for NaN values
    if np.isnan(true_center).any() or np.isnan(pred_center).any():
        return None, None
    
    # Round coordinates to 3 decimal places
    true_center = np.round(true_center, 3)
    pred_center = np.round(pred_center, 3)
    
    # Calculate deviations
    delta_x = pred_center[0] - true_center[0]
    delta_y = pred_center[1] - true_center[1]
    delta_elements = len(true_indices) -len(pred_indices)
    
    deviation_metrics = (delta_x, delta_y, delta_elements)
    coordinates = (true_center, pred_center, true_coords, pred_coords)
    
    return deviation_metrics, coordinates


def plot_random_deviations(mesh_obj, true_perms, predicted_perms, num_samples=10, threshold=0.5, 
                      save=False, fpath='', fname=''):
   
    random_indices = np.random.choice(len(true_perms), size=num_samples, replace=False)
    cols = 5
    rows = int(np.ceil(num_samples / cols))
    
    subplot_size = 4 
    fig, axes = plt.subplots(rows, cols, figsize=(cols * subplot_size, rows * subplot_size))
    axes = axes.flatten()
    
    for i, idx in enumerate(random_indices):
        ax = axes[i]
        true_perm = true_perms[idx]
        predicted_perm = predicted_perms[idx]
        
        # Verwende die neue Funktion
        metrics, coordinates = compute_evaluation_metrics(
            mesh_obj, true_perm, predicted_perm, threshold=threshold
        )
        
        if metrics is None:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
            continue
            
        delta_x, delta_y, _ = metrics
        (true_center, pred_center, true_coords, pred_coords) = coordinates
        
        ax.grid()
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.set_aspect('equal')
        
        ax.scatter(mesh_obj.node[:, 0], mesh_obj.node[:, 1], color="grey", s=0.1, label="Mesh")
        ax.scatter(true_coords[:, 0], true_coords[:, 1], color="C1", s=2, label="True")
        ax.scatter(pred_coords[:, 0], pred_coords[:, 1], color="C2", s=2, label="Pred")
        ax.scatter(true_center[0], true_center[1], marker="x", color="C3", s=50, label="Center True")
        ax.scatter(pred_center[0], pred_center[1], marker="x", color="C4", s=50, label="Center Pred")
        ax.set_title(f"Sample {idx}\nΔx={delta_x:.2f}, Δy={delta_y:.2f}")
        ax.legend(fontsize=6)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    if save:
        full_path = fpath + fname
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {full_path}")
    

def plot_random_recon_examples(mesh_obj, true_perm, pred_perm, num_samples):
    random_indices = np.random.choice(len(true_perm), size=num_samples, replace=False)
    
    cols = 5
    rows = 2
    fig = plt.figure(figsize=(4*cols, 4*rows))
    
    pts = mesh_obj.node
    tri = mesh_obj.element
    x, y = pts[:, 0], pts[:, 1]
    
    # Calculate triangle centers for plotting true values
    tri_centers = np.mean(pts[tri], axis=1)
    
    for i, idx in enumerate(random_indices):
        true_values = true_perm[idx].flatten()
        pred_values = pred_perm[idx].flatten()
    
        ax = plt.subplot(rows, cols, i + 1)
        
        # Plot predicted values with tripcolor
        im_pred = ax.tripcolor(x, y, tri, pred_values,
                             shading="flat", edgecolor="k", alpha=1.0)
        
        # Overlay true values as scatter points at triangle centers
        threshold = 0.5
        mask = true_values > threshold
        if np.any(mask):
             ax.scatter(tri_centers[mask, 0], tri_centers[mask, 1], 
                  color='sandybrown',  # Saddlebrown oder alternativ 'brown'
                  alpha=0.15, s=10)
        
        ax.set_aspect("equal")
        ax.set_ylim([-1.2, 1.2])
        ax.set_xlim([-1.2, 1.2])
        ax.set_title(f"Sample {idx}")
        plt.colorbar(im_pred, ax=ax)
        
    plt.show()


def plot_deviations_x_y(true_perms, pred_perms, mesh_obj, threshold=0.5, save=False, 
                       fpath='', fname='x_y_deviation.pdf', figsize=(8, 8), limits=(-1, 1)):
 
    # Compute deviations for all samples
    x_deviations = []
    y_deviations = []
    
    for true_perm, pred_perm in zip(true_perms, pred_perms):
        metrics, _ = compute_evaluation_metrics(mesh_obj, true_perm, pred_perm, threshold)
        if metrics is not None:
            delta_x, delta_y, _ = metrics
            x_deviations.append(delta_x)
            y_deviations.append(delta_y)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x-Abweichung': x_deviations,
        'y-Abweichung': y_deviations
    })
    
    # Style settings
    plt.rcParams.update({'font.family': 'Serif'})
    sns.set(font_scale=1.5, font='Serif', style='whitegrid')
    
    # Create joint plot
    g = sns.jointplot(
        data=df,
        x='x-Abweichung',
        y='y-Abweichung',
        kind='kde',
        xlim=limits,
        ylim=limits,
        height=figsize[0]
    )
    
    # Enhance the plot
    g.plot_joint(sns.kdeplot, fill=True, levels=50, cmap='viridis')
    
    # Add mean lines
    x_mean = np.mean(x_deviations)
    y_mean = np.mean(y_deviations)
    g.ax_joint.axvline(x=x_mean, color='r', linestyle='--', alpha=0.5)
    g.ax_joint.axhline(y=y_mean, color='r', linestyle='--', alpha=0.5)
    
    # Add statistics text box
    #stats_text = (f"μx = {x_mean:.3f}\n"
    #             f"μy = {y_mean:.3f}\n"
     #            f"σx = {np.std(x_deviations):.3f}\n"
      #           f"σy = {np.std(y_deviations):.3f}")
    #g.ax_joint.text(0.02, 0.98, stats_text,
                  # transform=g.ax_joint.transAxes,
                  # verticalalignment='top',
                  # bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save:
        plt.tight_layout()
        full_path = fpath + fname
        g.savefig(full_path)
        print(f"Plot saved to: {full_path}")
    
    return g

def plot_deviations_perm(true_perms, pred_perms, mesh_obj, threshold=0.5, 
                        save=False, fpath='', fname='perm_deviation.pdf', binwidth=10):
    
    # Compute element deviations for all samples
    perm_deviations = []
    
    for true_perm, pred_perm in zip(true_perms, pred_perms):
        metrics, _ = compute_evaluation_metrics(mesh_obj, true_perm, pred_perm, threshold)
        if metrics is not None:
            _, _, delta_perm = metrics
            perm_deviations.append(delta_perm)
    
    # Style settings
    plt.rcParams.update({'font.family': 'Serif'})
    sns.set(font_scale=2, font='Serif')
    
    # Create figure
    plt.figure(figsize=(7, 7))
    plt.autoscale()
    
    # Create histogram plot
    p = sns.histplot(data=perm_deviations,
                    binwidth=binwidth,
                    kde=True,
                    color='navy',
                    alpha=0.6)  
    
    # Calculate statistics
    mean_dev = np.mean(perm_deviations)
    std_dev = np.std(perm_deviations)
    st_fe = 2840  # Total number of finite elements
    percent_dev = (mean_dev/st_fe) * 100
    
    # Add vertical mean line
    p.axvline(x=mean_dev, color='red', linestyle=':', alpha=0.8)
    
    # Set labels
    p.set_xlabel("Abweichende Elemente")
    p.set_ylabel("Anzahl")
    fig = p.get_figure()
    
    if save:
        # Save plot
        plt.tight_layout()
        plt.savefig(fpath + fname)
        
        # Save statistics to text file
        stats_filename = fpath + 'perm_deviation_stats.txt'
        with open(stats_filename, 'w') as f:
            f.write(f'Mittlere Perm-Abweichung: {round(mean_dev)} [FE]\n')
            f.write(f'Standardabweichung: {round(std_dev, 1)} [FE]\n')
            f.write(f'Prozentuale Abweichung: {round(percent_dev, 2)} [%]\n')
        print(f"Statistics saved to: {stats_filename}")
    
    return fig

from PyPDF2 import PdfWriter, PdfReader

def merge_plots_to_pdf(x_y_plot, perm_plot, output_path, output_name='merged_analysis.pdf'):
    
   # Save individual plots to PDFs
   x_y_pdf = output_path + 'temp_xy.pdf'
   perm_pdf = output_path + 'temp_perm.pdf'
   
   x_y_plot.savefig(x_y_pdf)
   perm_plot.savefig(perm_pdf)
   
   # Create merged PDF
   output_pdf = output_path + output_name
   
   # PDF Reader für beide Eingabe-PDFs
   reader_left = PdfReader(x_y_pdf)
   reader_right = PdfReader(perm_pdf)
   
   # Neue PDF erstellen
   writer = PdfWriter()
   
   # Erste Seite von jeder PDF nehmen
   left_page = reader_left.pages[0]
   right_page = reader_right.pages[0]
   
   # Seiten zum Writer hinzufügen
   writer.add_page(left_page)
   writer.add_page(right_page)
   
   # Als neue PDF speichern
   with open(output_pdf, 'wb') as output_file:
       writer.write(output_file)
   
   # Temporäre Dateien löschen
   import os
   os.remove(x_y_pdf)
   os.remove(perm_pdf)
   
   print(f"Merged analysis plots saved as: {output_pdf}")

def run_deviation_analysis(gamma_true, gamma_pred_test, mesh_obj, fpath='Abbildungen/', base_fname='sim_few_unbalanced_interpol_reconstruct'):
    """
    Runs complete deviation analysis including plots and PDF merging.
    
    Parameters
    ----------
    gamma_true : array
        True permittivity values
    gamma_pred_test : array
        Predicted permittivity values
    mesh_obj : object
        Mesh object containing node and element information
    fpath : str
        Base path for saving files (default: 'Abbildungen/')
    base_fname : str
        Base filename for all outputs (default: 'sim_few_unbalanced_interpol_reconstruct')
    """
    # Generate specific filenames
    position_fname = f'position_{base_fname}_deviations.pdf'
    perm_fname = f'perm_{base_fname}_deviations.pdf'
    combined_fname = f'combined_{base_fname}_deviations.pdf'
    
    # Plot position deviations direkt mit den rohen Daten
    g = plot_deviations_x_y(gamma_true, gamma_pred_test, mesh_obj, 
                           save=True,
                           fpath=fpath,
                           fname=position_fname)
    
    # Plot permittivity deviations direkt mit den rohen Daten
    fig = plot_deviations_perm(gamma_true, gamma_pred_test, mesh_obj,
                              save=True,
                              fpath=fpath,
                              fname=perm_fname,
                              binwidth=15)
    
    # Merge PDFs
    merge_plots_to_pdf(g, fig, fpath, combined_fname)
    
    print(f"Analysis complete. Final merged PDF saved as: {fpath}{combined_fname}")