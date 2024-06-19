# Tracking Climate Change With Satellites and Artificial Intelligence
We are the Duke Bass Connections Team 2023-2024, working on tracking climate change with satellite imagery and artificial intelligence.

As part of our research, we performed a literature review to ensure that we were doing novel work. 
- Benchmark dataset analysis: https://docs.google.com/spreadsheets/d/1TnvYGRL3HyH6l7IbzJB0baEvP2q-HOGEnmVfD6m0QGc/edit?usp=sharing
- Relevant paper analysis to determine current conditions of the research space: https://docs.google.com/document/d/1iYtCwHk5dyqfae9rxr8D1R6RqMqDejktVpoB4F77KXo/edit?usp=sharing

Our experiments contain two parts: CLIP zero-shot and few-shot learning; Visual-large language model zero-shot testing;


Visual-large language model:
Since running a visual-large language model requires lots of GPU memory, we recommend running our code in the Google Colab with high-performance GPU runtime.
- Llava: Run the src/Llava.ipynb
- LHRS-Bot: Run the src/LHRS_BOT.ipynb


CLIP few-shot learning:
We applied Cafo and Tip-adapter.
- Cafo: We mainly use the code from https://github.com/OpenGVLab/CaFo. Please check https://github.com/DaryLu0v0/CaFo_Bass.
