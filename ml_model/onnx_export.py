import onnxruntime as ort
import numpy as np
import chess
import sys

# Mapping from chess pieces to channels
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

def predict(fen, model_path="model.onnx"):
    ort_session = ort.InferenceSession(model_path)
    board_tensor = encode_fen(fen)
    input_tensor = np.expand_dims(board_tensor, axis=0).astype(np.float32)  # shape (1, 12, 8, 8)

    inputs = {"board": input_tensor}
    outputs = ort_session.run(None, inputs)
    return float(outputs[0][0][0])  # scalar output

if __name__ == "__main__":
    fen = sys.argv[1]  # pass FEN as command line argument
    score = predict(fen)
    print(f"Predicted score: {score:.4f}")
