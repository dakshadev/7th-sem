# Execution constraint — user clarification

The user reaffirmed **free Colab/Kaggle for GPU training**. Do not install local CUDA packages or launch local GPU training/smoke runs. Local development and bounded CPU correctness tests are allowed; use the CPU-only PyTorch environment. The parent stopped the CUDA-wheel download before PyTorch was installed; no local model training had run at that point.

GPU-specific tests may exist but should be skipped on the CPU development environment and executed later in the user's free cloud runtime. Do not claim cloud execution without actual runtime evidence. Provide portable requirements and exact tested CPU commands plus a clearly marked cloud execution handoff. Person 2 retains ownership of the shared notebook unless the user authorizes otherwise.
