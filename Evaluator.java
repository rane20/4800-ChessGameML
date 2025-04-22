import ai.onnxruntime.*;
import java.util.*;
import chesspresso.position.Position;

public class Evaluator {
    private OrtEnvironment env;
    private OrtSession session;

    private static final Map<Character, Integer> PIECE_TO_INDEX = new HashMap<>();
static {
    PIECE_TO_INDEX.put('P', 0); PIECE_TO_INDEX.put('N', 1);
    PIECE_TO_INDEX.put('B', 2); PIECE_TO_INDEX.put('R', 3);
    PIECE_TO_INDEX.put('Q', 4); PIECE_TO_INDEX.put('K', 5);
    PIECE_TO_INDEX.put('p', 6); PIECE_TO_INDEX.put('n', 7);
    PIECE_TO_INDEX.put('b', 8); PIECE_TO_INDEX.put('r', 9);
    PIECE_TO_INDEX.put('q', 10); PIECE_TO_INDEX.put('k', 11);
}


    public Evaluator(String modelPath) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(modelPath, new OrtSession.SessionOptions());
    }

    public float evaluateFEN(String fen) throws OrtException {
        float[][][][] input = new float[1][12][8][8];
        Position position = new Position(fen);

        for (int sq = 0; sq < 64; sq++) {
            int piece = position.getPiece(sq);
            if (piece == chesspresso.Chess.NO_PIECE) continue;
        
            char symbol = chesspresso.Chess.pieceToChar(piece);  // This gives 'P', 'n', etc.
        
            int row = 7 - (sq / 8);
            int col = sq % 8;
            int index = PIECE_TO_INDEX.getOrDefault(symbol, -1);
        
            if (index >= 0) input[0][index][row][col] = 1.0f;
        }
        
        

        OnnxTensor tensor = OnnxTensor.createTensor(env, input);
        OrtSession.Result result = session.run(Collections.singletonMap("board", tensor));
        float[][] output = (float[][]) result.get(0).getValue();
        return output[0][0];
    }

    public void close() throws OrtException {
        session.close();
        env.close();
    }
}

