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
from PIL import Image  

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
    

def plot_random_recon_examples(mesh_obj, true_perm, pred_perm, num_samples, 
                             save=False, fpath='', fname=''):
    
    random_indices = np.random.choice(len(true_perm), size=num_samples, replace=False)
    
    cols = 5
    rows = 5
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
        ax.set_xticks([])
        ax.set_yticks([])
        im_ped = ax.tripcolor(x, y, tri, pred_values, shading="flat", edgecolor ='none', linewidth=0.001, alpha=1.0, cmap='viridis')

        # Plot true values
        threshold = 0.5
        mask = true_values > threshold
        if np.any(mask):
            for j in np.where(mask)[0]:
                triangle = tri[j]
                triangle_pts = pts[triangle]
                ax.fill(triangle_pts[:, 0], triangle_pts[:, 1], 
                       color='goldenrod', alpha=0.5,
                      edgecolor='w',
                     linewidth=0.001)
        
        # Set limits to exactly match the frame
        ax.set_ylim([-1, 1])
        ax.set_xlim([-1, 1])
        # Force the aspect ratio to be exactly square and fill the frame
        ax.set_aspect('equal', adjustable='box', anchor='C')
    
    plt.tight_layout()
    
    if save:
        # Create full paths for both PNG and PDF
        base_path = os.path.splitext(fpath + fname)[0]  # Remove any existing extension
        png_path = base_path + '.png'
        pdf_path = base_path + '.pdf'
        
        try:
            # Save as PNG first
            plt.savefig(png_path, bbox_inches='tight', dpi=300)
            
            # Convert PNG to PDF
            image = Image.open(png_path)
            # Convert to RGB if necessary (in case the image is RGBA)
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            
            image.save(pdf_path, 'PDF', resolution=300.0)
            
            print(f"Plots saved as:\nPNG: {png_path}\nPDF: {pdf_path}")
            
            # Optionally remove the PNG file if you don't want to keep it
            os.remove(png_path)
            
        except Exception as e:
            print(f"Error saving files: {str(e)}")
    
    plt.show()



def plot_deviations_x_y(true_perms, pred_perms, mesh_obj, threshold=0.5, save=False, 
                       fpath='', fname='', figsize=(8, 8), limits=(-1, 1)):
 
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
    
    stats_filename = fpath + fname + '_stats.txt'
        
    # Save statistics to text file
    with open(stats_filename, 'w') as f:
        f.write(f'μx = {x_mean:.3f}\n')
        f.write(f'μy = {y_mean:.3f}\n')
        f.write(f'σx = {np.std(x_deviations):.3f}\n')
        f.write(f'σy = {np.std(y_deviations):.3f}\n')
    print(f"Statistics saved to: {stats_filename}")

    if save:
        plt.tight_layout()
        full_path = fpath + fname
        g.savefig(full_path)
        print(f"Plot saved to: {full_path}")
    
    return g

def plot_deviations_perm(true_perms, pred_perms, mesh_obj, threshold=0.5, 
                        save=False, fpath='', fname='', binwidth=10):
    
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
        
        stats_filename = fpath + fname + '_stats.txt'
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
    

def normalize_angle(angle):
    """Convert angle from [-180, 180] to [0, 360] range"""
    return (angle + 360) % 360

def compute_polar_metrics(mesh_obj, true_perm, pred_perm, threshold=0.5):
    """
    Computes polar coordinate metrics (radius and angle) for the predictions vs ground truth
    
    Args:
        mesh_obj: Mesh object containing node and element information
        true_perm: Ground truth permittivity values
        pred_perm: Predicted permittivity values
        threshold: Threshold value for binary segmentation
    
    Returns:
        tuple: (r_true, r_pred, theta_true, theta_pred) arrays containing the radii and angles
    """
    
    # Make sure we use only the matching timesteps
    min_timesteps = min(true_perm.shape[0], pred_perm.shape[0])
    true_perm = true_perm[:min_timesteps]
    pred_perm = pred_perm[:min_timesteps]
    
    metrics_list = []
    coordinates_list = []
    
    for t in range(min_timesteps):
        metrics, coords = compute_evaluation_metrics(mesh_obj, true_perm[t], pred_perm[t], threshold)
        if metrics is not None:
            metrics_list.append(metrics)
            coordinates_list.append(coords)
    
    r_true = []
    r_pred = []
    theta_true = []
    theta_pred = []
    
    def compute_polar_metrics(mesh_obj, true_perm, pred_perm, threshold=0.5):
        """
        Computes polar coordinate metrics (radius and angle) for the predictions vs ground truth
        """
        # Bisheriger Code bleibt gleich bis zur Winkelberechnung
        min_timesteps = min(true_perm.shape[0], pred_perm.shape[0])
        true_perm = true_perm[:min_timesteps]
        pred_perm = pred_perm[:min_timesteps]
        
        metrics_list = []
        coordinates_list = []
        
        for t in range(min_timesteps):
            metrics, coords = compute_evaluation_metrics(mesh_obj, true_perm[t], pred_perm[t], threshold)
            if metrics is not None:
                metrics_list.append(metrics)
                coordinates_list.append(coords)
        
        r_true = []
        r_pred = []
        theta_true = []
        theta_pred = []
        
        for coords in coordinates_list:
            true_center, pred_center, true_coords, pred_coords = coords
            
            # Radien bleiben unverändert
            r_t = np.sqrt(np.mean(true_coords[:,0]**2 + true_coords[:,1]**2))
            r_p = np.sqrt(np.mean(pred_coords[:,0]**2 + pred_coords[:,1]**2))
            
            # Winkelberechnung ohne Normalisierung
            theta_t = np.degrees(np.arctan2(np.mean(true_coords[:,1]), 
                                          np.mean(true_coords[:,0])))
            theta_p = np.degrees(np.arctan2(np.mean(pred_coords[:,1]), 
                                          np.mean(pred_coords[:,0])))
            
            r_true.append(r_t)
            r_pred.append(r_p)
            theta_true.append(theta_t)
            theta_pred.append(theta_p)
        
        # Arrays erstellen
        theta_true = np.array(theta_true)
        theta_pred = np.array(theta_pred)
        
        # Unwrapping durchführen
        theta_true_unwrapped = np.unwrap(np.radians(theta_true))
        theta_pred_unwrapped = np.unwrap(np.radians(theta_pred))
        
        # Zurück zu Grad konvertieren
        theta_true_unwrapped = np.degrees(theta_true_unwrapped)
        theta_pred_unwrapped = np.degrees(theta_pred_unwrapped)
        
        # Offset anpassen, damit die Werte nicht bei 360° beginnen
        # Finden des minimalen Werts und Verschieben aller Werte
        min_theta = min(np.min(theta_true_unwrapped), np.min(theta_pred_unwrapped))
        theta_true_unwrapped -= min_theta
        theta_pred_unwrapped -= min_theta
        
        return np.array(r_true), np.array(r_pred), theta_true_unwrapped, theta_pred_unwrapped

def plot_polar_metrics(r_true, r_pred, theta_true, theta_pred, save=False, fpath='', fname=''):
    """
    Creates plots comparing the true and predicted radii and angles with customized styling
    """
    # Set white background
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    
        # Time points starting at 4
    t = np.arange(4, len(r_true) + 4)
    max_t = 5000
    mid_t = max_t // 2
    quarter_t = mid_t // 2  # 1250
    three_quarter_t = mid_t + quarter_t  # 3750
    
    # First plot (radius)
    ax1.plot(t, r_true, color='orangered', linestyle='-', linewidth=1)
    ax1.plot(t, r_pred, color='steelblue', linestyle='-', linewidth=1)
    
    # Customize x-axis ticks and grid
    ax1.set_xticks([0, quarter_t, mid_t, three_quarter_t, max_t])
    ax1.set_xticklabels(['0', '1250', '2500', '3750', '5000'], fontsize=15)
    ax1.set_xlim(0, max_t)
    
    # Customize y-axis ticks and grid
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=15)
    
    ax1.set_xlabel('n', fontsize=15)
    ax1.set_ylabel('Radius', fontsize=15)
    #ax1.legend(frameon=True, facecolor='white', edgecolor='none')
    #ax1.grid(True, color='lightgray', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Second plot (angle)
    ax2.plot(t, theta_true, color='orangered', linestyle='-', linewidth=1)
    ax2.plot(t, theta_pred, color='steelblue', linestyle='-', linewidth=1)
    
    # Customize x-axis ticks and grid
    ax1.set_xticks([0, quarter_t, mid_t, three_quarter_t, max_t])
    ax1.set_xticklabels(['0', '1250', '2500', '3750', '5000'], fontsize=15)
    ax1.set_xlim(0, max_t)
    
    ax2.set_xlabel('t', fontsize=15)
    ax2.set_ylabel('Winkel θ (°)', fontsize=15)
    ax2.set_ylim(0, 360)
    ax2.legend(frameon=True, facecolor='white', edgecolor='none')
    ax2.grid(True, color='lightgray', linewidth=0.5, alpha=0.5)
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save:
        # Save as PNG first
        png_path = fpath + fname + '.png'
        pdf_path = fpath + fname + '.pdf'
        
        plt.savefig(png_path, bbox_inches='tight', dpi=300)
        
        # Convert to PDF
        from PIL import Image
        image = Image.open(png_path)
        image.save(pdf_path, 'PDF', resolution=300.0)
        print(f"Plots saved as:\nPNG: {png_path}\nPDF: {pdf_path}")

def run_deviation_analysis(gamma_true, gamma_pred_test, mesh_obj, fpath='Abbildungen/', base_fname='sim_few_unbalanced_interpol_reconstruct'):

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