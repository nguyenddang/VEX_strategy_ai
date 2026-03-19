from evaluator.evaluator import Evaluator

if __name__ == "__main__":
    evaluator = Evaluator(checkpoint_path='checkpoints_2', n_games_per_matchup=750, n_workers=90)
    evaluator.run()