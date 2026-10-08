#!/usr/bin/env python3
"""Bounding-box size statistics (COCO small/medium/large, percentiles) and histograms of a MOT dataset GT.

    python tools/bbox_stats.py data/real --out_plots outputs/bbox_plots
"""

import argparse
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def process_file(file_path):
    """Extrae las áreas, anchos y altos de un archivo MOTChallenge."""
    areas, widths, heights = [], [], []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) < 6:
                    continue 
                
                try:
                    width = float(parts[4])
                    height = float(parts[5])
                except ValueError:
                    continue
                
                if width <= 0 or height <= 0:
                    continue
                    
                areas.append(width * height)
                widths.append(width)
                heights.append(height)
    except Exception as e:
        print(f"Error al leer {file_path}: {e}")
        
    return areas, widths, heights

def print_statistics(name, areas, widths, heights, is_global=False):
    """Calcula e imprime las estadísticas, percentiles y umbrales."""
    if not areas:
        print(f"No hay datos válidos para {name}.")
        return

    # Convertir a arrays de numpy para cálculos rápidos
    arr_areas = np.array(areas)
    arr_widths = np.array(widths)
    arr_heights = np.array(heights)
    total = len(arr_areas)

    # Básicas
    min_area, max_area, avg_area = np.min(arr_areas), np.max(arr_areas), np.mean(arr_areas)
    min_w, max_w, avg_w = np.min(arr_widths), np.max(arr_widths), np.mean(arr_widths)
    min_h, max_h, avg_h = np.min(arr_heights), np.max(arr_heights), np.mean(arr_heights)

    # Umbrales estándar de COCO
    # Small < 32^2 (1024) | Medium 32^2 - 96^2 (1024-9216) | Large > 96^2 (9216)
    small_mask = arr_areas < 1024
    medium_mask = (arr_areas >= 1024) & (arr_areas <= 9216)
    large_mask = arr_areas > 9216

    num_small = np.sum(small_mask)
    num_medium = np.sum(medium_mask)
    num_large = np.sum(large_mask)

    # Percentiles
    p_areas = np.percentile(arr_areas, [25, 50, 75, 90, 95])
    p_widths = np.percentile(arr_widths, [50, 90])
    p_heights = np.percentile(arr_heights, [50, 90])

    if is_global:
        print("\n" + "=" * 80)
        print(f"  {name}")
        print("=" * 80)
    else:
        print(f"\n[{name}]")

    print(f"Total de Bounding Boxes: {total:,}")
    print(f"  ÁREA  | Mínima: {min_area:8.1f} | Promedio: {avg_area:8.1f} | Máxima: {max_area:8.1f}")
    print(f"  ANCHO | Mínimo: {min_w:8.1f} | Promedio: {avg_w:8.1f} | Máximo: {max_w:8.1f}")
    print(f"  ALTO  | Mínimo: {min_h:8.1f} | Promedio: {avg_h:8.1f} | Máximo: {max_h:8.1f}\n")

    print(f"  > Clasificación de Tamaños (Umbrales COCO):")
    print(f"    - Pequeños (< 32x32 px)     : {num_small:,} ({(num_small/total)*100:.1f}%)")
    print(f"    - Medianos (32x32 - 96x96)  : {num_medium:,} ({(num_medium/total)*100:.1f}%)")
    print(f"    - Grandes  (> 96x96 px)     : {num_large:,} ({(num_large/total)*100:.1f}%)\n")

    print(f"  > Percentiles de ÁREA:")
    print(f"    25%: {p_areas[0]:.1f} | 50% (Mediana): {p_areas[1]:.1f} | 75%: {p_areas[2]:.1f} | 90%: {p_areas[3]:.1f} | 95%: {p_areas[4]:.1f}")
    print(f"  > Medianas de Dimensión: Ancho={p_widths[0]:.1f} px | Alto={p_heights[0]:.1f} px")
    print(f"  > El 90% de los objetos miden menos de: Ancho={p_widths[1]:.1f} px | Alto={p_heights[1]:.1f} px")
    
    if not is_global:
        print("-" * 80)

def plot_distributions(name, areas, widths, heights, output_dir):
    """Genera y guarda histogramas y gráficos de dispersión de los bboxes."""
    if not areas:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(areas, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_title(f'Distribución de Áreas - {name}')
    ax1.set_xlabel('Área (px²)')
    ax1.set_ylabel('Frecuencia')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.scatter(widths, heights, alpha=0.3, color='coral', s=10)
    ax2.set_title(f'Ancho vs Alto - {name}')
    ax2.set_xlabel('Ancho (px)')
    ax2.set_ylabel('Alto (px)')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    safe_name = name.replace(" ", "_")
    save_path = os.path.join(output_dir, f"{safe_name}_dist.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Calcula estadísticas de bboxes y genera gráficas para un dataset MOT.")
    parser.add_argument("dataset_dir", help="Ruta al directorio principal del dataset (ej. datasets/real_all/)")
    parser.add_argument("--out_plots", default="bbox_plots", help="Carpeta donde se guardarán las gráficas (default: bbox_plots)")
    args = parser.parse_args()

    search_pattern = os.path.join(args.dataset_dir, "*", "gt", "gt.txt")
    gt_files = sorted(glob.glob(search_pattern))

    if not gt_files:
        print(f"No se encontraron archivos gt.txt usando el patrón: {search_pattern}")
        return

    os.makedirs(args.out_plots, exist_ok=True)

    all_areas = []
    all_widths = []
    all_heights = []

    print(f"Iniciando análisis en: {args.dataset_dir}")
    print(f"Archivos encontrados: {len(gt_files)}")
    print(f"Las gráficas se guardarán en: ./{args.out_plots}/")
    print("-" * 80)

    for file_path in gt_files:
        seq_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        
        areas, widths, heights = process_file(file_path)
        
        print_statistics(seq_name, areas, widths, heights, is_global=False)
        plot_distributions(seq_name, areas, widths, heights, args.out_plots)
        
        all_areas.extend(areas)
        all_widths.extend(widths)
        all_heights.extend(heights)

    if all_areas:
        print_statistics("RESUMEN GLOBAL DEL DATASET", all_areas, all_widths, all_heights, is_global=True)
        plot_distributions("GLOBAL_DATASET", all_areas, all_widths, all_heights, args.out_plots)
    else:
        print("\nNo se encontraron bounding boxes válidos en todo el dataset.")

if __name__ == "__main__":
    main()