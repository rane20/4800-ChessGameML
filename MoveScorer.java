import ai.onnxruntime.*;
import chesspresso.move.IllegalMoveException;
import chesspresso.move.Move;
import chesspresso.position.Position;

public class MoveScorer {
    private Evaluator evaluator;

    public static class ScoredMove {
        public String moveSAN;
        public String nextFEN;
        public float score;
        public float delta;
        public float quality;

        public ScoredMove(String moveSAN, String nextFEN, float score, float delta, float quality) {
            this.moveSAN = moveSAN;
            this.nextFEN = nextFEN;
            this.score = score;
            this.delta = delta;
            this.quality = quality;
        }
    }

    public MoveScorer(String modelPath) throws OrtException {
        this.evaluator = new Evaluator(modelPath);
    }

    public ScoredMove findBestMove(Position position, int fromSqi) throws OrtException {
        String currentFEN = position.getFEN();
        float currentEval = evaluator.evaluateFEN(currentFEN);
    
        short[] legalMoves = position.getAllMoves();
        ScoredMove best = new ScoredMove("none", currentFEN, currentEval, 0, 0);
        float bestQuality = -1.0f;
    
        for (short move : legalMoves) {
            if (Move.getFromSqi(move) != fromSqi) continue; // 👈 only consider clicked piece
    
            try {
                position.doMove(move);
                String nextFEN = position.getFEN();
                float nextEval = evaluator.evaluateFEN(nextFEN);
                float delta = nextEval - currentEval;
                float quality = sigmoid(delta);
                String san = Move.getString(move);
                position.undoMove();
    
                if (quality > bestQuality) {
                    bestQuality = quality;
                    best = new ScoredMove(san, nextFEN, nextEval, delta, quality);
                }
            } catch (IllegalMoveException e) {
                position.undoMove(); // Safe cleanup
            }
        }
    
        if (best.moveSAN.equals("none")) {
            System.out.println("[AI DEBUG] No better move found from this piece.");
        }
        return best;
        
    }
    

    public void close() throws OrtException {
        evaluator.close();
    }

    private float sigmoid(float x) {
        return (float)(1.0 / (1.0 + Math.exp(-x)));
    }
    public Evaluator getEvaluator() {
        return evaluator;
    }
    
}

