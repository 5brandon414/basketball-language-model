# References

## Direct methodological lineage (framing or methods this work builds on)

1. Simpson et al. (2022). "Seq2Event: Learning the Language of Soccer
   using Transformer-based Match Event Prediction." First transformer
   next-event model for soccer; the "match as language" framing.
2. Mendes-Neves et al. (2024). "Towards a Foundation Large Events Model
   for Soccer." Machine Learning (Springer).
   https://link.springer.com/article/10.1007/s10994-024-06606-y
3. "EventGPT: Capturing Player Impact from Team Action Sequences Using
   GPT-Based Framework" (2025). Closest single relative: decoder-only,
   player-conditioned, counterfactual swaps. https://arxiv.org/abs/2512.17266
4. "Transformer-Based Baseball Modeling for Pitch Outcome Prediction and
   Strategy Optimization" (SSAC). Source of the next-event evaluation
   format adopted here; scope difference is the contribution: single
   pitch outcome vs full-game generative simulation.
   https://www.sloansportsconference.com/research-papers/transformer-based-baseball-modeling-for-pitch-outcome-prediction-and-strategy-optimization

## Simulation and win-probability ancestors (positioned against)

5. Oh, Keshri, Iyengar (SSAC 2015). "Graphical Model for Basketball
   Match Simulation." The closest ancestor at this conference: full-game
   NBA simulation via a handcrafted probabilistic graphical model —
   possession-aggregate and handcrafted structure vs learned event-level
   generation with identity, rotations, and calibration.
   https://www.sloansportsconference.com/research-papers/graphical-model-for-baskeball-match-simulation
6. Vracar, Strumbelj, Kononenko (2016). "Modeling basketball play-by-play
   data" (possession-level Markov simulation). Expert Systems with
   Applications. The handcrafted ancestor of the learned approach.
7. Stern (1994). "A Brownian Motion Model for the Progress of Sports
   Scores." JASA. Root of the in-game win-probability lineage the
   halftime-lead baseline belongs to.
8. Cervone, D'Amour, Bornn, Goldsberry (SSAC 2014). "POINTWISE:
   Predicting Points and Valuing Decisions in Real Time." EPV; the
   ancestor of querying a model from any game state.

## Differentiation (related basketball deep models, not generative full-game)

9. Alcorn & Nguyen (2021). "baller2vec: A Multi-Entity Transformer for
   Multi-Agent Spatiotemporal Modeling." Trajectory-level, discriminative.
   https://arxiv.org/abs/2102.03291
10. "A Deep Learning Based Approach for Live Win Probability in NBA Games
    Using Play-by-Play Events and Compact Game State" (2026).
    Discriminative win-prob from PBP; no generation, single output.
    https://link.springer.com/chapter/10.1007/978-3-032-27272-0_7

## Venue precedents

11. "CoachAI+ Badminton Environment: Realistic Badminton Game Simulator"
    (SSAC 2025). A full game simulator as a Sloan finalist.
    https://www.sloansportsconference.com/research-papers/coachai-badminton-environment-realistic-badminton-game-simulator-for-enhancing-player-performance
12. "(batter|pitcher)2vec: Statistic-Free Talent Modeling With Neural
    Player Embeddings" (SSAC 2018). Player embeddings at Sloan; won the
    2018 RPC.
    https://www.sloansportsconference.com/research-papers/batter-pitcher-2vec-statistic-free-talent-modeling-with-neural-player-embeddings
