import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import chess
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Map chess pieces to tensor indices
PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def encode_fen(fen):
    board_tensor = np.zeros((12, 8, 8), dtype=np.float32)
    board = chess.Board(fen)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            index = PIECE_TO_INDEX[piece.symbol()]
            row = 7 - (square // 8)
            col = square % 8
            board_tensor[index, row, col] = 1
    return board_tensor

class ChessDatasetFromDF(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen = self.data.iloc[idx]['fen']
        score = self.data.iloc[idx]['score'] / 100.0  # normalize
        board_tensor = encode_fen(fen)
        return torch.tensor(board_tensor), torch.tensor([score], dtype=torch.float32)

class EvaluationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

def train(csv_path="chess_data.csv", epochs=30, batch_size=32):
    # Load and sample data
    full_df = pd.read_csv(csv_path, names=["fen", "score"], header=None)
    df = full_df.sample(n=80000, random_state=42).reset_index(drop=True) #80,000 random samples

    # Split train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create datasets and dataloaders
    train_dataset = ChessDatasetFromDF(train_df)
    test_dataset = ChessDatasetFromDF(test_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    model = EvaluationModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for boards, scores in train_loader:
            optimizer.zero_grad()
            preds = model(boards)
            loss = loss_fn(preds, scores)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for boards, scores in test_loader:
                preds = model(boards)
                loss = loss_fn(preds, scores)
                total_test_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_train_loss:.4f} | Test Loss: {total_test_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "fen_evaluator_model.pt")

    # Export to ONNX
    export_to_onnx("fen_evaluator_model.pt", "model.onnx")

    # Evaluation
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for boards, scores in test_loader:
            preds = model(boards).squeeze().numpy()
            targets = scores.squeeze().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)

    r2 = r2_score(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    print(f"\n R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    # Sample output
    print("\n Sample Predictions:")
    for i in range(5):
        print(f"  Predicted: {all_preds[i]:.2f} | Actual: {all_targets[i]:.2f}")

def export_to_onnx(model_path="fen_evaluator_model.pt", onnx_path="model.onnx"):
    model = EvaluationModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dummy_input = torch.randn(1, 12, 8, 8)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["board"],
        output_names=["evaluation"],
        dynamic_axes={"board": {0: "batch_size"}, "evaluation": {0: "batch_size"}},
        opset_version=11
    )

if __name__ == "__main__":
    train()




