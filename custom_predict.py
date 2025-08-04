import argparse
import json
import torch
from MolNexTR import molnextr
import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def predict_folder(model, folder_path, return_atoms_bonds=False, return_confidence=False):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    predictions = []

    # Check if CUDA is enabled
    cuda_enabled = torch.cuda.is_available()
    if cuda_enabled:
        print("CUDA is enabled.")
    else:
        print("CUDA is not available. Using CPU.")

    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(folder_path, image_file)
        output = model.predict_final_results(
            image_path, return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)
        output['image_file'] = image_file
        predictions.append(output)

        # Log output every 100 predictions
        if idx % 100 == 0:
            print(f"Completed predictions for {idx} images")

    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, required=True)
    parser.add_argument('--image_folder', type=str, default=None, required=True)
    parser.add_argument('--return_atoms_bonds', action='store_true')
    parser.add_argument('--return_confidence', action='store_true')
    parser.add_argument('--output_csv', type=str, default='predictions.csv')
    args = parser.parse_args()

    device = torch.device('cuda')
    model = molnextr(args.model_path, device)
    predictions = predict_folder(
        model, args.image_folder, 
        return_atoms_bonds=args.return_atoms_bonds, 
        return_confidence=args.return_confidence)
    
    # Convert prediction results to a DataFrame
    df_predictions = pd.DataFrame(predictions)
    
    # Save to CSV
    df_predictions.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")
