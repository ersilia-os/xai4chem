# cli.py
import argparse
from .train import train
from .inference import infer

class XAI4ChemCLI:
    def __call__(self):
        parser = argparse.ArgumentParser(
            description="XAI4Chem CLI - A command-line interface for training and inference with XAI4Chem.",
        )

        subparsers = parser.add_subparsers(dest='command')

        # Training command
        train_parser = subparsers.add_parser('train', help='Train a model with the given input data.')
        train_parser.add_argument('--input_file', type=str, required=True, help='Path to the CSV file containing input data (must include "smiles" and "activity" columns).')
        train_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the trained model and evaluation reports.')
        train_parser.add_argument('--representation', type=str, required=True, choices=['datamol_descriptor', 'rdkit_descriptor', 'mordred_descriptor', 'morgan_fingerprint', 'rdkit_fingerprint'],
                                  help='Type of molecular representation to use. Options are: datamol_descriptor, rdkit_descriptor, mordred_descriptor, morgan_fingerprint, rdkit_fingerprint.')
        train_parser.set_defaults(func=train)

        # Inference command
        infer_parser = subparsers.add_parser('infer', help='Make predictions with a trained model.')
        infer_parser.add_argument('--input_file', type=str, required=True, help='Path to the CSV file containing input data (must include "smiles" column).')
        infer_parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the saved model file.')
        infer_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the prediction results.')
        infer_parser.set_defaults(func=infer)

        args = parser.parse_args()
        args.func(args)

if __name__ == '__main__':
    cli = XAI4ChemCLI()
    cli()
