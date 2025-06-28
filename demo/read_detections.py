#!/usr/bin/env python3
"""
Utility script to read and work with saved detection results from YOLO-World.

Usage:
    python demo/read_detections.py demo_outputs/bus_detections.json
    python demo/read_detections.py demo_outputs/ --all
"""

import json
import argparse
import os
import csv
from pathlib import Path

def read_detection_json(json_path):
    """Read detection results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def print_detection_summary(data):
    """Print a summary of detections."""
    print(f"Image: {data['image_name']}")
    print(f"Total detections: {data['total_detections']}")
    print("-" * 50)
    
    for i, detection in enumerate(data['detections'], 1):
        bbox = detection['bbox']
        print(f"Detection {i}:")
        print(f"  Class: {detection['class_name']} (ID: {detection['class_id']})")
        print(f"  Confidence: {detection['confidence']:.3f}")
        print(f"  Bounding Box: ({bbox['x1']:.1f}, {bbox['y1']:.1f}) to ({bbox['x2']:.1f}, {bbox['y2']:.1f})")
        print(f"  Size: {bbox['width']:.1f} x {bbox['height']:.1f}")
        print(f"  Center: ({bbox['center_x']:.1f}, {bbox['center_y']:.1f})")
        print()

def filter_detections_by_class(data, class_names):
    """Filter detections by class names."""
    filtered_detections = []
    for detection in data['detections']:
        if detection['class_name'] in class_names:
            filtered_detections.append(detection)
    
    return {
        'image_name': data['image_name'],
        'image_path': data['image_path'],
        'detections': filtered_detections,
        'total_detections': len(filtered_detections)
    }

def filter_detections_by_confidence(data, min_confidence):
    """Filter detections by minimum confidence score."""
    filtered_detections = []
    for detection in data['detections']:
        if detection['confidence'] >= min_confidence:
            filtered_detections.append(detection)
    
    return {
        'image_name': data['image_name'],
        'image_path': data['image_path'],
        'detections': filtered_detections,
        'total_detections': len(filtered_detections)
    }

def get_detection_statistics(data):
    """Get statistics about detections."""
    if not data['detections']:
        return {}
    
    confidences = [d['confidence'] for d in data['detections']]
    areas = [d['bbox']['width'] * d['bbox']['height'] for d in data['detections']]
    classes = [d['class_name'] for d in data['detections']]
    
    stats = {
        'total_detections': len(data['detections']),
        'avg_confidence': sum(confidences) / len(confidences),
        'max_confidence': max(confidences),
        'min_confidence': min(confidences),
        'avg_area': sum(areas) / len(areas),
        'max_area': max(areas),
        'min_area': min(areas),
        'unique_classes': list(set(classes)),
        'class_counts': {cls: classes.count(cls) for cls in set(classes)}
    }
    
    return stats

def export_filtered_csv(data, output_path):
    """Export filtered detection results to CSV."""
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'class_name', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2', 'width', 'height', 'center_x', 'center_y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for detection in data['detections']:
            row = {
                'image_name': data['image_name'],
                'class_name': detection['class_name'],
                'class_id': detection['class_id'],
                'confidence': detection['confidence'],
                'x1': detection['bbox']['x1'],
                'y1': detection['bbox']['y1'],
                'x2': detection['bbox']['x2'],
                'y2': detection['bbox']['y2'],
                'width': detection['bbox']['width'],
                'height': detection['bbox']['height'],
                'center_x': detection['bbox']['center_x'],
                'center_y': detection['bbox']['center_y']
            }
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description='Read and analyze YOLO-World detection results')
    parser.add_argument('input', help='JSON file path or directory containing JSON files')
    parser.add_argument('--all', action='store_true', help='Process all JSON files in directory')
    parser.add_argument('--filter-class', nargs='+', help='Filter by class names')
    parser.add_argument('--min-confidence', type=float, default=0.0, help='Minimum confidence threshold')
    parser.add_argument('--stats', action='store_true', help='Show detection statistics')
    parser.add_argument('--export-csv', help='Export filtered results to CSV file')
    
    args = parser.parse_args()
    
    # Get JSON files to process
    json_files = []
    if os.path.isfile(args.input):
        json_files = [args.input]
    elif os.path.isdir(args.input) and args.all:
        json_files = list(Path(args.input).glob("*_detections.json"))
    else:
        print(f"Error: {args.input} is not a valid file or use --all for directory processing")
        return
    
    all_data = []
    
    for json_file in json_files:
        print(f"\n{'='*60}")
        print(f"Processing: {json_file}")
        print('='*60)
        
        # Read detection data
        data = read_detection_json(json_file)
        
        # Apply filters
        if args.filter_class:
            data = filter_detections_by_class(data, args.filter_class)
            print(f"Filtered by classes: {args.filter_class}")
        
        if args.min_confidence > 0:
            data = filter_detections_by_confidence(data, args.min_confidence)
            print(f"Filtered by confidence >= {args.min_confidence}")
        
        # Show results
        print_detection_summary(data)
        
        if args.stats:
            stats = get_detection_statistics(data)
            if stats:
                print("Detection Statistics:")
                print(f"  Average confidence: {stats['avg_confidence']:.3f}")
                print(f"  Confidence range: {stats['min_confidence']:.3f} - {stats['max_confidence']:.3f}")
                print(f"  Average area: {stats['avg_area']:.1f} pixels")
                print(f"  Area range: {stats['min_area']:.1f} - {stats['max_area']:.1f} pixels")
                print(f"  Unique classes: {', '.join(stats['unique_classes'])}")
                print(f"  Class counts: {stats['class_counts']}")
                print()
        
        all_data.append(data)
    
    # Export to CSV if requested
    if args.export_csv:
        combined_data = {
            'image_name': 'combined',
            'image_path': 'multiple',
            'detections': [],
            'total_detections': 0
        }
        
        for data in all_data:
            combined_data['detections'].extend(data['detections'])
        
        combined_data['total_detections'] = len(combined_data['detections'])
        export_filtered_csv(combined_data, args.export_csv)
        print(f"Exported {combined_data['total_detections']} detections to {args.export_csv}")

if __name__ == '__main__':
    main() 