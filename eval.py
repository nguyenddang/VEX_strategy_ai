from evaluator.evaluator import Evaluator

if __name__ == "__main__":
    evaluator = Evaluator(checkpoint_path='checkpoints_1', n_games_per_matchup=500, n_workers=10)
    evaluator.run()