import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

def load_results(json_path):
    """Load and parse the final results JSON file."""
    try:
        print(f"Attempting to load results from: {json_path}")
        if not os.path.exists(json_path):
            print(f"Error: File not found at {json_path}")
            return None
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            print(f"Successfully loaded data with {len(data)} addresses")
            return data
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        return None

def extract_metrics_for_address(address_data):
    """Extract metrics for a single address."""
    data = []
    
    print(f"\nTotal images in address data: {len(address_data)}")  # Debug print
    
    # Iterate through each image in the address
    for image_path, image_data in address_data.items():
        try:
            if 'response' not in image_data:
                print(f"Warning: No 'response' field for {image_path}")
                continue
            
            metrics = image_data['response']
            if isinstance(metrics, str):
                metrics = json.loads(metrics)
            metrics['image_path'] = image_path
            data.append(metrics)
        except Exception as e:
            print(f"Error processing metrics for {image_path}: {str(e)}")
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    print(f"Successfully processed images: {len(df)}")  # Debug print
    return df

def visualize_ranking(df, metric_name, file_store_path, address, num_images=20):
    """
    Visualize top and bottom images for a given metric in 4 rows of 5 images each.
    Top 10 images in first two rows, bottom 10 images in last two rows.
    """
    if df is None or metric_name not in df.columns:
        print(f"Error: Cannot visualize ranking for metric '{metric_name}'")
        return None
    
    print(f"\nVisualizing ranking for metric: {metric_name}")
    
    # Sort by metric
    df_sorted = df.sort_values(by=metric_name, ascending=False)
    
    # Get top and bottom images
    top_images = df_sorted.head(num_images // 2)  # Get top 10
    bottom_images = df_sorted.tail(num_images // 2)  # Get bottom 10
    
    # Create subplot with 4 rows, 5 columns
    fig, axes = plt.subplots(4, 5, figsize=(25, 24))  # Further increased height for more spacing
    plt.subplots_adjust(top=0.92, bottom=0.08)  # Adjust top and bottom margins
    fig.suptitle(f'{address}\nTop and Bottom Images by {metric_name}', fontsize=20, y=0.96)  # Increased title font size
    
    def process_images(image_set, start_row, row_label):
        """Process a set of images across two rows"""
        for idx, (_, row) in enumerate(image_set.iterrows()):
            # Calculate row and column indices
            r = start_row + (idx // 5)  # Move to next row after 5 images
            c = idx % 5  # Cycle through columns 0-4
            
            img_path = os.path.join(file_store_path, row['image_path'])
            print(f"Processing {row_label} image {idx+1}: {img_path}")
            
            try:
                if not os.path.exists(img_path):
                    print(f"Image not found: {img_path}")
                    raise FileNotFoundError(f"Image not found: {img_path}")
                
                img = Image.open(img_path)
                axes[r, c].imshow(img)
                axes[r, c].axis('off')
                
                # Calculate serial number based on position
                # For top images: 1-10
                # For bottom images: 11-20
                serial_num = idx + 1 if row_label == "top" else idx + 11
                title = f'#{serial_num}\nScore: {row[metric_name]:.2f}'
                axes[r, c].set_title(title, fontsize=20, pad=12)  # Increased font size and padding
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                axes[r, c].text(0.5, 0.5, 'Image not found', ha='center', fontsize=14)
                axes[r, c].axis('off')
    
    # Process top 10 images (first two rows)
    process_images(top_images, 0, "top")
    
    # Process bottom 10 images (last two rows)
    process_images(bottom_images, 2, "bottom")
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])  # Adjust layout while preserving space for the title
    return fig

def analyze_address(address, address_data, file_store_path, output_dir):
    """Analyze and visualize rankings for a single address."""
    print(f"\nAnalyzing address: {address}")
    
    # Create output directory for this address
    address_output_dir = output_dir / address
    address_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics for this address
    df = extract_metrics_for_address(address_data)
    if df is None:
        print(f"No valid metrics found for address: {address}")
        return
    
    print(f"Found {len(df)} images with metrics")
    print("Available columns:", df.columns.tolist())
    
    # Get all numeric metrics (excluding image_path)
    metrics = df.select_dtypes(include=['float64', 'int64']).columns
    print(f"Numeric metrics to analyze: {metrics.tolist()}")  # Debug print
    
    # Create visualizations for each metric
    for metric in metrics:
        print(f"\nProcessing metric: {metric} for {address}")
        
        # Save complete rankings to CSV first
        rankings_df = df[['image_path', metric]].sort_values(by=metric, ascending=False)
        rankings_path = address_output_dir / f'rankings_{metric}.csv'
        rankings_df.to_csv(rankings_path, index=False)
        print(f"Saved complete rankings ({len(rankings_df)} images) to {rankings_path}")
        
        # Create visualization
        fig = visualize_ranking(df, metric, file_store_path, address)
        if fig is not None:
            output_path = address_output_dir / f'ranking_{metric}.png'
            try:
                fig.savefig(output_path)
                print(f"Saved visualization to {output_path}")
                plt.close(fig)
            except Exception as e:
                print(f"Error saving figure: {str(e)}")

def main():
    # Setup paths
    results_path = Path('my_test_data/file_store/final_results.json')
    file_store_path = Path('my_test_data/file_store')
    output_dir = Path('analysis/output')
    
    print("\nStarting analysis...")
    print(f"Results path: {results_path}")
    print(f"File store path: {file_store_path}")
    print(f"Output directory: {output_dir}")
    
    # Create main output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Created output directory successfully")
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")
        return

    # Load results
    results = load_results(results_path)
    if results is None:
        return
    
    # Process each address separately
    for address, address_data in results.items():
        analyze_address(address, address_data, file_store_path, output_dir)

if __name__ == "__main__":
    main() 