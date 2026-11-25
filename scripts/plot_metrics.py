#!/usr/bin/env python3
"""visualize training metrics"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


def parse_log_file(log_path: str) -> pd.DataFrame:
    """parse training log file."""
    with open(log_path, 'r') as f:
        content = f.read()
    
    # extract scalar values
    pattern = r'(\w+): ([\d.]+)'
    matches = re.findall(pattern, content)
    
    data = {}
    for key, val in matches:
        data[key] = data.get(key, []) + [float(val)]
    
    return pd.DataFrame(data)


def plot_training_curves(metrics_path: str, output: str = 'training_curves.png'):
    """plot training metrics from tensorboard events or csv."""
    # try to load as csv first
    csv_path = Path(metrics_path) / 'metrics.csv'
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        # parse from text
        log_files = list(Path(metrics_path).glob('*.log'))
        if not log_files:
            # try events
            import os
            events_files = list(Path(metrics_path).rglob('events.*'))
            if events_files:
                print(f"converting tensorboard events from {events_files[0]}...")
                from tensorboard.backend.event_processing import event_accumulator
                ea = event_accumulator.EventAccumulator(str(events_files[0]))
                ea.Reload()
                tags = ea.Tags()['scalars']
                data = {tag: [] for tag in tags}
                
                for tag in tags:
                    for event in ea.Scalars(tag):
                        data[tag].append((event.step, event.value))
                
                df = pd.DataFrame({k: [v[1] for v in vals] for k, vals in data.items()})
            else:
                print("no metrics found")
                return
        else:
            df = parse_log_file(str(log_files[0]))
    
    # plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # loss
    if 'mse' in df.columns:
        axes[0, 0].plot(df['mse'].rolling(100).mean())
        axes[0, 0].set_title('noise mse (smoothed)')
        axes[0, 0].set_xlabel('step')
        axes[0, 0].set_ylabel('mse')
    
    # learning rate
    if 'lr' in df.columns:
        axes[0, 1].plot(df['lr'])
        axes[0, 1].set_title('learning rate')
        axes[0, 1].set_xlabel('step')
    
    # sample quality (if tracked)
    if 'fid' in df.columns:
        axes[1, 0].plot(df['fid'])
        axes[1, 0].set_title('FID score')
        axes[1, 0].set_xlabel('step')
    
    # gradient norm
    if 'grad_norm' in df.columns:
        axes[1, 1].plot(df['grad_norm'])
        axes[1, 1].set_title('gradient norm')
        axes[1, 1].set_xlabel('step')
    
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"saved to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='checkpoints/logs/',
                        help='tensorboard logs directory')
    parser.add_argument('--output', type=str, default='training_curves.png',
                        help='output image path')
    
    args = parser.parse_args()
    plot_training_curves(args.log_dir, args.output)


if __name__ == '__main__':
    main()