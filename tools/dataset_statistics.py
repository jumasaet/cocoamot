#!/usr/bin/env python3
"""Dataset statistics (sequences, frames, boxes, ids, duration) of a MOTChallenge dataset.

    python tools/dataset_statistics.py data/real --fps 20
"""

import os
import glob
import configparser
import argparse



def get_dataset_metrics(dataset_path, fps):
    total_bboxes = 0
    total_frames = 0
    total_unique_ids = 0
    total_sequences = 0
    resolutions = set()
    
    # Buscar todos los directorios dentro de la ruta que contengan 'seqinfo.ini'
    seq_dirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    print(f"Procesando el dataset en: {dataset_path}")
    print("-" * 60)
    print(f"{'Secuencia':<15} | {'Frames':<8} | {'BBoxes':<8} | {'IDs Únicos':<10} | {'Resolución'}")
    print("-" * 60)

    for seq_name in seq_dirs:
        seq_path = os.path.join(dataset_path, seq_name)
        seqinfo_path = os.path.join(seq_path, 'seqinfo.ini')
        gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
        
        # Validar que sea una secuencia MOT válida
        if not os.path.exists(seqinfo_path) or not os.path.exists(gt_path):
            continue
            
        total_sequences += 1
        
        # 1. Extraer Resolución y Frames de seqinfo.ini
        config = configparser.ConfigParser()
        config.read(seqinfo_path)
        
        try:
            w = config['Sequence']['imWidth']
            h = config['Sequence']['imHeight']
            seq_frames = int(config['Sequence']['seqLength'])
            resolutions.add(f"{w}x{h}")
        except KeyError:
            print(f"Error leyendo {seqinfo_path}, verifica su formato.")
            continue
            
        total_frames += seq_frames
        
        # 2. Extraer BBoxes e IDs de gt.txt
        seq_bboxes = 0
        seq_ids = set()
        
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    seq_bboxes += 1
                    # En MOT15/16/17/20, el ID del objeto es la segunda columna (índice 1)
                    obj_id = parts[1]
                    seq_ids.add(obj_id)
                    
        total_bboxes += seq_bboxes
        # Los IDs son únicos por secuencia, por lo que sumamos la cantidad de cada secuencia
        num_seq_ids = len(seq_ids)
        total_unique_ids += num_seq_ids
        
        print(f"{seq_name:<15} | {seq_frames:<8} | {seq_bboxes:<8} | {num_seq_ids:<10} | {w}x{h}")

    print("-" * 60)
    
    # 3. Calcular Duración Total
    total_duration_sec = total_frames / fps
    total_duration_min = total_duration_sec / 60

    # 4. Reporte Final
    print("\n" + "=" * 30)
    print("MÉTRICAS GLOBALES DEL DATASET")
    print("=" * 30)
    print(f"Total Secuencias : {total_sequences}")
    print(f"Resolución(es)   : {', '.join(resolutions)}")
    print(f"Total Frames     : {total_frames}")
    print(f"Total BBoxes     : {total_bboxes}")
    print(f"Total IDs Únicos : {total_unique_ids} (Acumulados)")
    print(f"FPS Asignados    : {fps}")
    print(f"Duración Total   : {total_duration_sec:.2f} segundos ({total_duration_min:.2f} minutos)")
    print("=" * 30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calcula métricas de un dataset en formato MOTChallenge.")
    parser.add_argument('path', type=str, help="Dataset root with <seq>/seqinfo.ini and <seq>/gt/gt.txt")
    parser.add_argument('--fps', type=float, default=20.0, help="Cuadros por segundo (FPS) del dataset")
    args = parser.parse_args()
    
    get_dataset_metrics(args.path, args.fps)