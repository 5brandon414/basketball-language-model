# References

## A. Direct methodological neighbors (cite and differentiate)

The "match as language" line; the basketball instantiation, rotations,
and calibrated distributions are the differentiation.

1. Simpson et al. (2022). "Seq2Event: Learning the Language of Soccer
   using Transformer-based Match Event Prediction." First transformer
   next-event model for soccer.
2. Mendes-Neves et al. (2024). "Towards a Foundation Large Events Model
   for Soccer." Machine Learning (Springer).
   https://link.springer.com/article/10.1007/s10994-024-06606-y
3. "A Scalable Approach for Unified Large Events Models in Soccer" (2025).
   https://link.springer.com/chapter/10.1007/978-3-032-06129-4_21
4. "EventGPT: Capturing Player Impact from Team Action Sequences Using
   GPT-Based Framework" (2025). Closest single relative: decoder-only,
   player-conditioned, counterfactual swaps. https://arxiv.org/abs/2512.17266
5. "Modeling Matches as Language: A Generative Transformer Approach for
   Counterfactual Player Valuation in Football" (2026).
   https://arxiv.org/abs/2603.15212

Basketball-adjacent deep models (differentiate: trajectories/discriminative,
not full-game event generation).

6. Alcorn & Nguyen (2021). "baller2vec: A Multi-Entity Transformer for
   Multi-Agent Spatiotemporal Modeling." https://arxiv.org/abs/2102.03291
7. Alcorn & Nguyen (2021). "baller2vec++." https://arxiv.org/abs/2104.11980
8. "NBA2Vec: Dense Feature Representations of NBA Players" (2023).
   https://arxiv.org/abs/2302.13386
9. "A Deep Learning Based Approach for Live Win Probability in NBA Games
   Using Play-by-Play Events and Compact Game State" (2026).
   https://link.springer.com/chapter/10.1007/978-3-032-27272-0_7
10. Sicilia et al. (2019). "DeepHoops: Evaluating Micro-Actions in
    Basketball Using Deep Feature Representations." KDD.

Generative simulation ancestors.

11. Vracar, Strumbelj, Kononenko (2016). "Modeling basketball play-by-play
    data" (possession-level Markov simulation). Expert Systems with
    Applications. The handcrafted ancestor of the learned approach.
12. "SportsNGEN: Sustained Generation of Multi-player Sports Gameplay"
    (2024). https://arxiv.org/abs/2403.12977
13. Stern (1994). "A Brownian Motion Model for the Progress of Sports
    Scores." JASA. Root of the in-game win-probability lineage.

Baseline lineage (for the ridge and minutes baselines).

14. Sill (2010). "Improved NBA Adjusted +/- Using Regularization and
    Out-of-Sample Testing." SSAC. The RAPM paper; direct ancestor of the
    ridge baseline.
15. DARKO (Medvedovsky), EPM (Snarr): public player-projection systems;
    industry-standard chain exemplars. https://www.darko.app/

## B. Same competition (Sloan RPC) precedents

Basketball track canon.

16. Cervone, D'Amour, Bornn, Goldsberry (SSAC 2014). "POINTWISE:
    Predicting Points and Valuing Decisions in Real Time." The canonical
    in-game value paper; EPV is the spiritual ancestor of any-state
    querying.
17. "Estimating Positional Plus-Minus in the NBA" (SSAC 2023).
    https://www.sloansportsconference.com/research-papers/estimating-positional-plus-minus-in-the-nba
18. "Estimating NBA Team Shot Selection Efficiency from Aggregations of
    True, Continuous Shot Charts" (SSAC 2024).
    https://www.sloansportsconference.com/research-papers/estimating-nba-team-shot-selection-efficiency-from-aggregations-of-true-continuous-shot-charts-a-generalized-additive-model-approach
19. "Deep Reinforcement Learning for NBA Player Valuation" (SSAC 2026
    finalist, Jenkins).

Sequence/generative precedents at Sloan.

20. "CoachAI+ Badminton Environment: Realistic Badminton Game Simulator"
    (SSAC 2025). A full game simulator as a Sloan finalist.
    https://www.sloansportsconference.com/research-papers/coachai-badminton-environment-realistic-badminton-game-simulator-for-enhancing-player-performance
21. "(batter|pitcher)2vec: Statistic-Free Talent Modeling With Neural
    Player Embeddings" (SSAC 2018). Player embeddings at Sloan; won the
    2018 RPC.
    https://www.sloansportsconference.com/research-papers/batter-pitcher-2vec-statistic-free-talent-modeling-with-neural-player-embeddings
22. "You Cannot Do That Ben Stokes: Dynamically Predicting Shot Type in
    Cricket Using a Personalized Deep Neural Network" (SSAC 2020).
    Personalized next-event prediction at Sloan.
    https://www.sloansportsconference.com/research-papers/you-cannot-do-that-ben-stokes-dynamically-predicting-shot-type-in-cricket-using-a-personalized-deep-neural-network
23. "Learning Contextual Event Embeddings to Predict Player Performance
    in the MLB" (SSAC 2023). Event-sequence embeddings at Sloan.
    https://www.sloansportsconference.com/research-papers/learning-contextual-event-embeddings-to-predict-player-performance-in-the-mlb
24. "Bhostgusters: Realtime Interactive Play Sketching with Synthesized
    NBA Defenses" (SSAC 2018). Generative NBA modeling at Sloan.
    https://www.sloansportsconference.com/research-papers/bhostgusters-realtime-interactive-play-sketching-with-synthesized-nba-defenses
25. "Data-Driven Ghosting using Deep Imitation Learning" (SSAC 2017).
    Generating counterfactual player behavior; a famous Sloan paper.
    https://www.sloansportsconference.com/research-papers/data-driven-ghosting-using-deep-imitation-learning

Additional Sloan precedents:

26. Oh, Keshri, Iyengar (SSAC 2015). "Graphical Model for Basketball
    Match Simulation." The closest ancestor at this conference: full-game
    NBA simulation via a handcrafted probabilistic graphical model.
    Differentiate: possession-aggregate + handcrafted structure vs
    learned event-level generation with identity, rotations, calibration.
    https://www.sloansportsconference.com/research-papers/graphical-model-for-baskeball-match-simulation
27. "Transformer-Based Baseball Modeling for Pitch Outcome Prediction and
    Strategy Optimization" (SSAC). The transformer-baseball precedent and the
    source of the next-event evaluation format we adopt; scope difference is
    our contribution: single pitch outcome vs full-game generative simulation.
    https://www.sloansportsconference.com/research-papers/transformer-based-baseball-modeling-for-pitch-outcome-prediction-and-strategy-optimization
