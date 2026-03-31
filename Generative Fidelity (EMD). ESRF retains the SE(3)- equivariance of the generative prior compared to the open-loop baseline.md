## Generative Fidelity (EMD)
ESRF retains the SE(3)-equivariance of the generative prior compared to the open-loop baseline.

| Method              | Laptop (EMD) ↓ | Mug (EMD) ↓ | Bowl (EMD) ↓ | Pencil (EMD) ↓ | Cup (EMD) ↓ | Knife (EMD) ↓ | Hammer (EMD) ↓ | Spoon (EMD) ↓ |
|---------------------|----------------|-------------|--------------|----------------|-------------|---------------|----------------|---------------|
| EquiGraspFlow (EGF) | 0.3420         | 0.4936      | 0.3131       | 0.2881         | 0.3382      | 0.3647        | 0.4304         | 0.3787        |
| **Ours (ESRF)**     | **0.3486**     | **0.4861**  | **0.2867**   | **0.3063**     | **0.3352**  | **0.3689**    | **0.4381**     | **0.3623**    |
