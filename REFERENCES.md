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
   format adopted here.
   https://www.sloansportsconference.com/research-papers/transformer-based-baseball-modeling-for-pitch-outcome-prediction-and-strategy-optimization

## Simulation and win-probability ancestors (positioned against)

5. Oh, Keshri, Iyengar (SSAC 2015). "Graphical Model for Basketball
   Match Simulation." Full-game NBA simulation from a handcrafted
   possession-aggregate graphical model; the closest ancestor of the
   learned event-level generation here.
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
    Player Embeddings" (SSAC 2018). Learned player embeddings at Sloan.
    https://www.sloansportsconference.com/research-papers/batter-pitcher-2vec-statistic-free-talent-modeling-with-neural-player-embeddings

## Added after the prior-art review (September 2026)

- Bhat, Huang, Rodriguez. Learning Stochastic Models for Basketball Substitutions
  from Play-by-Play Data. MLSA workshop at ECML-PKDD, 2015 (CEUR Vol-1970, paper 8).
  Continuous-time Markov chain over five-man lineups; full-game simulation with
  handcrafted rates, including a lineup-removal counterfactual. The closest prior
  basketball model in which the model itself chooses who is on the floor.
- Mendes-Neves, Meireles, Mendes-Moreira. Forecasting Events in Soccer Matches
  Through Language. arXiv:2402.06820, 2024. Full-match event rollouts to outcome
  probabilities; team-level, no player identity.
- Hong et al. Modeling Matches as Language: A Generative Transformer Approach for
  Counterfactual Player Valuation in Football (ScoutGPT). arXiv:2603.15212, 2026.
  Decoder-only event model; lineups are user-specified and fixed during generation.
- Yeung, Sit, Fujii. Transformer-based neural marked spatio-temporal point process
  model for football match events (NMSTPP). arXiv:2302.09276; Applied Intelligence, 2025.
- Lieder. NBA Game Simulation Using RNN and Adversarial Networks, Part 1. Medium,
  2021/22. Word-level LSTM over a ~14-symbol NBA event vocabulary sampling whole
  games; no players, lineups, or forecasts. Cited as public prior art.
- Yeh, Rice, Dubin. Evaluating Real-Time Probabilistic Forecasts with Application to
  NBA Outcome Prediction. The American Statistician 76(3), 2022.
- Gneiting, Raftery. Strictly Proper Scoring Rules, Prediction, and Estimation.
  JASA, 2007. (Coverage and CRPS conventions.)
- Hughes. An Approximate Dynamic Programming Approach to Determine the Optimal
  Substitution Strategy for Basketball. PhD thesis, George Mason University, 2017.
  Prescriptive lineup decisions under endurance uncertainty; not generative.
- Romero, Mashayekhi, Lai, Van Roy et al. Next-Event Prediction in Soccer: Assessing
  the Impact of Team and Player Information. Data Mining for Sports, Springer, 2025.
