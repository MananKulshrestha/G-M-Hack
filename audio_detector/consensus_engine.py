import numpy as np

class ConsensusEngine:
    def __init__(self):
        pass

    def calculate_csi(self, mean, std, epsilon=0.01):
        """
        Calculates Consensus Strength Index (CSI).
        CSI = |Mean - 0.5| / (StdDev + Epsilon)
        """
        return abs(mean - 0.5) / (std + epsilon)

    def analyze_results(self, aggregated_scores):
        """
        aggregated_scores: dict {model_key: score} (Max score per model across chunks)
        """
        valid_scores = {k: v for k, v in aggregated_scores.items() if v is not None}
        
        if len(valid_scores) < 3:
            return {
                "verdict": "INCONCLUSIVE",
                "details": "Insufficient model responses (Quorum < 3)",
                "csi": 0.0,
                "mean": 0.0
            }

        # 1. Calculate Opinion Strength |Score - 0.5|
        opinion_strengths = {k: abs(v - 0.5) for k, v in valid_scores.items()}
        
        # 2. Select Top 3 'Vocal Majority'
        # We trust models that are Opinionated (high strength) more than confused ones (0.5)
        top_models = sorted(opinion_strengths, key=opinion_strengths.get, reverse=True)[:3]
        top_scores = [valid_scores[m] for m in top_models]
        
        # 3. Compute Statistics
        mu = np.mean(top_scores)
        sigma = np.std(top_scores)
        csi = self.calculate_csi(mu, sigma)

        # 4. Pessimistic Override Check
        # If any single trusted model screams FAKE (> 0.98), we flag it regardless of average
        pessimistic_flag = False
        highest_fake_score = max(valid_scores.values())
        if highest_fake_score > 0.95:
            pessimistic_flag = True

        # 5. Determine Verdict
        verdict = "UNCERTAIN"
        confidence = "LOW"
        
        # Logic Matrix
        if csi > 2.0:
            if mu > 0.6:
                verdict = "FAKE"
                confidence = "HIGH"
            elif mu < 0.4:
                verdict = "REAL"
                confidence = "HIGH"
        elif csi > 0.8:
            if mu > 0.5:
                verdict = "LIKELY FAKE" 
                confidence = "MEDIUM"
            else:
                verdict = "LIKELY REAL"
                confidence = "MEDIUM"
        
        # Conflict Handling
        if sigma > 0.25:
             confidence = "CONFLICTED"
             if pessimistic_flag:
                 verdict = "SUSPICIOUS (High Conflict)"
             else:
                 verdict = "INCONCLUSIVE (High Conflict)"

        return {
            "verdict": verdict,
            "confidence": confidence,
            "csi": round(csi, 2),
            "mean_score": round(mu, 4),
            "std_dev": round(sigma, 4),
            "top_contributors": top_models,
            "all_scores": valid_scores,
            "pessimistic_override": pessimistic_flag
        }