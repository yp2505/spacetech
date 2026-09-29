# Prompt for the OSG Technical Paper

Copy the following prompt into a ChatGPT Project after uploading the accompanying source bundle.

```text
Write a 5–6 page technical report titled:

“Orbital Salience Gating (OSG): Physics-Informed Episodic Memory Retrieval for an AI Satellite Swarm”

Use the uploaded SpaceTech source bundle as the only authority for implementation-specific claims. Use reliable external sources only for general orbital-mechanics and reinforcement-learning background. Prefer peer-reviewed papers, NASA publications, IEEE Xplore, and official documentation. Give an in-text citation and a working reference link for every external factual claim.

Do not claim that OSG is globally novel. Describe it as a “proposed physics-informed retrieval heuristic” unless a literature review establishes a narrower, defensible novelty claim. Do not invent experiments, metrics, results, source files, or implementation details.

Report requirements:

1. Abstract (150–200 words).
2. Introduction: satellite-swarm autonomy, reinforcement learning, and the need for episodic memory.
3. System context: explain the SpaceTech simulation, PPO/MAPPO, episodic memory, and the ARTEMIS flight-software role.
4. State the OSG objective: retrieve past episodes that are orbitally relevant, high quality, and recent enough to inform the current action.
5. Present and define the implemented formula:

   OSGScore_j = cosine_similarity(x_current, x_j)
                × sigmoid(R_j − R_mean)
                × exp(−beta × age_j)

6. Derive beta step by step from Kepler’s third law:

   T = 2π sqrt((R_earth + h)^3 / mu)
   exp(−beta × T_steps) = 0.5
   beta = ln(2) / T_steps

   Explain the design interpretation: one orbital period is the memory half-life.

7. Use the default 550 km LEO configuration and report the values implemented in the code:
   T ≈ 5730.13 seconds, T ≈ 95.50 minutes,
   T_steps ≈ 360.39, beta ≈ 0.001923349 per simulation step.
8. Explain exactly how OSG is now connected to PPO action selection:
   - OSG ranks stored episodes.
   - The top retrieved episodes are compressed into a four-value memory context:
     reward quality, collision rate, fuel-out rate, and retrieval confidence.
   - This context is appended to the 48-dimensional PPO local observation before model.predict().
   - The flight-software AI brain uses the same OSG context.
   - Completed simulation episodes now store orbital state and timestamp for later retrieval.
9. Map these claims to the relevant uploaded files and functions.
10. Include a cosine-only retrieval baseline and describe how the updated experiment is now closed-loop: retrieved context can affect PPO actions.
11. Treat matlab_export/osg_comparison_results.csv as a legacy, pre-integration result. It must not be used as evidence of the updated system’s performance. State that retraining and a new evaluation are required.
12. Discuss real-world relevance: recurring orbital conditions, eclipse transitions, energy management, anomaly response, and memory-assisted autonomy.
13. Give limitations and future work: retraining, repeated seeds, statistical tests, feature normalization, LEO/MEO/GEO tests, fixed- and learned-decay baselines, and safety validation.
14. Finish with a conclusion and a complete IEEE-style reference list.

Use professional, original academic language. Produce a Word-ready report with headings, equations, a table mapping equations to code files, and no unsupported claims.
```
