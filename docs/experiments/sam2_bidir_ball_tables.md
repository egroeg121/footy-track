### Held-out accuracy by anchor spacing

| clip | config | N | mode | n held-out | SAM2 absent | median err px | hit@5 | hit@10 | hit@20 | wrong-object (>50px, of present) |
|---|---|---|---|---|---|---|---|---|---|---|
| arsenal_example | base_plus/crop512/fp16/box | 10 | interp | 133 | 0% | 4.72 | 52% | 69% | 87% | 3% |
| arsenal_example | base_plus/crop512/fp16/box | 10 | fwd | 133 | 11% | 1.05 | 82% | 82% | 82% | 7% |
| arsenal_example | base_plus/crop512/fp16/box | 10 | bwd | 133 | 5% | 1.03 | 79% | 80% | 82% | 11% |
| arsenal_example | base_plus/crop512/fp16/box | 10 | nearest | 133 | 8% | 1.03 | 81% | 82% | 83% | 7% |
| arsenal_example | base_plus/crop512/fp16/box | 10 | scorew | 133 | 0% | 1.09 | 82% | 83% | 85% | 12% |
| arsenal_example | base_plus/crop512/fp16/box | 10 | hybrid | 133 | 0% | 1.09 | 82% | 83% | 85% | 12% |
| arsenal_example | base_plus/crop512/fp16/box | 25 | interp | 140 | 0% | 9.74 | 34% | 51% | 62% | 9% |
| arsenal_example | base_plus/crop512/fp16/box | 25 | fwd | 140 | 13% | 1.11 | 81% | 81% | 81% | 7% |
| arsenal_example | base_plus/crop512/fp16/box | 25 | bwd | 140 | 9% | 1.05 | 83% | 84% | 86% | 5% |
| arsenal_example | base_plus/crop512/fp16/box | 25 | nearest | 140 | 14% | 1.11 | 80% | 81% | 81% | 3% |
| arsenal_example | base_plus/crop512/fp16/box | 25 | scorew | 140 | 2% | 1.14 | 84% | 86% | 87% | 9% |
| arsenal_example | base_plus/crop512/fp16/box | 25 | hybrid | 140 | 0% | 1.15 | 84% | 86% | 88% | 9% |
| arsenal_example | base_plus/crop512/fp16/box | 50 | interp | 143 | 0% | 25.96 | 22% | 41% | 47% | 31% |
| arsenal_example | base_plus/crop512/fp16/box | 50 | fwd | 143 | 17% | 1.06 | 82% | 82% | 82% | 1% |
| arsenal_example | base_plus/crop512/fp16/box | 50 | bwd | 143 | 26% | 0.99 | 66% | 67% | 69% | 6% |
| arsenal_example | base_plus/crop512/fp16/box | 50 | nearest | 143 | 21% | 1.0 | 76% | 76% | 76% | 2% |
| arsenal_example | base_plus/crop512/fp16/box | 50 | scorew | 143 | 6% | 1.04 | 85% | 87% | 88% | 4% |
| arsenal_example | base_plus/crop512/fp16/box | 50 | hybrid | 143 | 0% | 1.09 | 85% | 87% | 88% | 8% |
| arsenal_example | base_plus/full/fp16/box | 10 | interp | 133 | 0% | 4.72 | 52% | 69% | 87% | 3% |
| arsenal_example | base_plus/full/fp16/box | 10 | fwd | 133 | 7% | 1.62 | 76% | 78% | 79% | 10% |
| arsenal_example | base_plus/full/fp16/box | 10 | bwd | 133 | 0% | 1.92 | 74% | 77% | 80% | 18% |
| arsenal_example | base_plus/full/fp16/box | 10 | nearest | 133 | 4% | 1.75 | 76% | 80% | 81% | 9% |
| arsenal_example | base_plus/full/fp16/box | 10 | scorew | 133 | 0% | 1.78 | 76% | 80% | 81% | 12% |
| arsenal_example | base_plus/full/fp16/box | 10 | hybrid | 133 | 0% | 1.78 | 76% | 80% | 81% | 12% |
| arsenal_example | base_plus/full/fp16/box | 25 | interp | 140 | 0% | 9.74 | 34% | 51% | 62% | 9% |
| arsenal_example | base_plus/full/fp16/box | 25 | fwd | 140 | 8% | 1.89 | 66% | 70% | 71% | 22% |
| arsenal_example | base_plus/full/fp16/box | 25 | bwd | 140 | 0% | 1.98 | 75% | 79% | 83% | 15% |
| arsenal_example | base_plus/full/fp16/box | 25 | nearest | 140 | 8% | 1.73 | 73% | 76% | 80% | 10% |
| arsenal_example | base_plus/full/fp16/box | 25 | scorew | 140 | 0% | 1.87 | 76% | 80% | 85% | 11% |
| arsenal_example | base_plus/full/fp16/box | 25 | hybrid | 140 | 0% | 1.87 | 76% | 80% | 85% | 11% |
| arsenal_example | base_plus/full/fp16/box | 50 | interp | 143 | 0% | 25.96 | 22% | 41% | 47% | 31% |
| arsenal_example | base_plus/full/fp16/box | 50 | fwd | 143 | 8% | 3.9 | 49% | 54% | 55% | 39% |
| arsenal_example | base_plus/full/fp16/box | 50 | bwd | 143 | 0% | 2.5 | 58% | 60% | 62% | 36% |
| arsenal_example | base_plus/full/fp16/box | 50 | nearest | 143 | 8% | 1.9 | 62% | 66% | 66% | 26% |
| arsenal_example | base_plus/full/fp16/box | 50 | scorew | 143 | 0% | 2.03 | 66% | 70% | 71% | 27% |
| arsenal_example | base_plus/full/fp16/box | 50 | hybrid | 143 | 0% | 2.03 | 66% | 70% | 71% | 27% |
| arsenal_example | small/crop512/fp16/box | 10 | interp | 133 | 0% | 4.72 | 52% | 69% | 87% | 3% |
| arsenal_example | small/crop512/fp16/box | 10 | fwd | 133 | 8% | 1.11 | 83% | 83% | 83% | 8% |
| arsenal_example | small/crop512/fp16/box | 10 | bwd | 133 | 5% | 1.23 | 79% | 80% | 82% | 11% |
| arsenal_example | small/crop512/fp16/box | 10 | nearest | 133 | 8% | 1.15 | 81% | 82% | 83% | 7% |
| arsenal_example | small/crop512/fp16/box | 10 | scorew | 133 | 0% | 1.21 | 82% | 83% | 85% | 12% |
| arsenal_example | small/crop512/fp16/box | 10 | hybrid | 133 | 0% | 1.21 | 82% | 83% | 85% | 12% |
| arsenal_example | small/crop512/fp16/box | 25 | interp | 140 | 0% | 9.74 | 34% | 51% | 62% | 9% |
| arsenal_example | small/crop512/fp16/box | 25 | fwd | 140 | 14% | 1.05 | 83% | 84% | 84% | 2% |
| arsenal_example | small/crop512/fp16/box | 25 | bwd | 140 | 9% | 1.23 | 84% | 84% | 86% | 5% |
| arsenal_example | small/crop512/fp16/box | 25 | nearest | 140 | 11% | 1.15 | 83% | 84% | 84% | 3% |
| arsenal_example | small/crop512/fp16/box | 25 | scorew | 140 | 6% | 1.16 | 85% | 86% | 88% | 5% |
| arsenal_example | small/crop512/fp16/box | 25 | hybrid | 140 | 0% | 1.22 | 88% | 90% | 93% | 5% |
| arsenal_example | small/crop512/fp16/box | 50 | interp | 143 | 0% | 25.96 | 22% | 41% | 47% | 31% |
| arsenal_example | small/crop512/fp16/box | 50 | fwd | 143 | 12% | 1.05 | 83% | 84% | 84% | 4% |
| arsenal_example | small/crop512/fp16/box | 50 | bwd | 143 | 26% | 1.09 | 66% | 68% | 69% | 6% |
| arsenal_example | small/crop512/fp16/box | 50 | nearest | 143 | 18% | 0.99 | 77% | 78% | 78% | 3% |
| arsenal_example | small/crop512/fp16/box | 50 | scorew | 143 | 4% | 1.08 | 85% | 87% | 88% | 7% |
| arsenal_example | small/crop512/fp16/box | 50 | hybrid | 143 | 0% | 1.1 | 85% | 87% | 88% | 8% |
| arsenal_example | small/full/fp16/box | 10 | interp | 133 | 0% | 4.72 | 52% | 69% | 87% | 3% |
| arsenal_example | small/full/fp16/box | 10 | fwd | 133 | 11% | 1.65 | 71% | 77% | 77% | 13% |
| arsenal_example | small/full/fp16/box | 10 | bwd | 133 | 0% | 1.99 | 68% | 71% | 77% | 18% |
| arsenal_example | small/full/fp16/box | 10 | nearest | 133 | 7% | 1.91 | 74% | 75% | 79% | 11% |
| arsenal_example | small/full/fp16/box | 10 | scorew | 133 | 0% | 1.99 | 75% | 77% | 81% | 14% |
| arsenal_example | small/full/fp16/box | 10 | hybrid | 133 | 0% | 1.99 | 75% | 77% | 81% | 14% |
| arsenal_example | small/full/fp16/box | 25 | interp | 140 | 0% | 9.74 | 34% | 51% | 62% | 9% |
| arsenal_example | small/full/fp16/box | 25 | fwd | 140 | 17% | 2.3 | 62% | 70% | 71% | 15% |
| arsenal_example | small/full/fp16/box | 25 | bwd | 140 | 1% | 2.1 | 67% | 81% | 84% | 13% |
| arsenal_example | small/full/fp16/box | 25 | nearest | 140 | 9% | 2.17 | 66% | 77% | 79% | 11% |
| arsenal_example | small/full/fp16/box | 25 | scorew | 140 | 1% | 2.2 | 70% | 83% | 86% | 11% |
| arsenal_example | small/full/fp16/box | 25 | hybrid | 140 | 0% | 2.24 | 70% | 83% | 86% | 11% |
| arsenal_example | small/full/fp16/box | 50 | interp | 143 | 0% | 25.96 | 22% | 41% | 47% | 31% |
| arsenal_example | small/full/fp16/box | 50 | fwd | 143 | 19% | 2.33 | 60% | 68% | 68% | 16% |
| arsenal_example | small/full/fp16/box | 50 | bwd | 143 | 0% | 3.18 | 52% | 55% | 57% | 42% |
| arsenal_example | small/full/fp16/box | 50 | nearest | 143 | 17% | 2.42 | 57% | 65% | 66% | 19% |
| arsenal_example | small/full/fp16/box | 50 | scorew | 143 | 0% | 3.08 | 60% | 70% | 72% | 27% |
| arsenal_example | small/full/fp16/box | 50 | hybrid | 143 | 0% | 3.08 | 60% | 70% | 72% | 27% |
| arsenal_example | tiny/crop512/fp16/box | 10 | interp | 133 | 0% | 4.72 | 52% | 69% | 87% | 3% |
| arsenal_example | tiny/crop512/fp16/box | 10 | fwd | 133 | 10% | 1.05 | 81% | 81% | 81% | 8% |
| arsenal_example | tiny/crop512/fp16/box | 10 | bwd | 133 | 2% | 1.18 | 80% | 80% | 82% | 14% |
| arsenal_example | tiny/crop512/fp16/box | 10 | nearest | 133 | 6% | 1.12 | 82% | 82% | 83% | 9% |
| arsenal_example | tiny/crop512/fp16/box | 10 | scorew | 133 | 0% | 1.14 | 83% | 83% | 85% | 12% |
| arsenal_example | tiny/crop512/fp16/box | 10 | hybrid | 133 | 0% | 1.14 | 83% | 83% | 85% | 12% |
| arsenal_example | tiny/crop512/fp16/box | 25 | interp | 140 | 0% | 9.74 | 34% | 51% | 62% | 9% |
| arsenal_example | tiny/crop512/fp16/box | 25 | fwd | 140 | 25% | 1.12 | 69% | 70% | 70% | 6% |
| arsenal_example | tiny/crop512/fp16/box | 25 | bwd | 140 | 3% | 1.15 | 83% | 84% | 86% | 10% |
| arsenal_example | tiny/crop512/fp16/box | 25 | nearest | 140 | 13% | 1.1 | 78% | 79% | 79% | 7% |
| arsenal_example | tiny/crop512/fp16/box | 25 | scorew | 140 | 2% | 1.1 | 84% | 86% | 87% | 9% |
| arsenal_example | tiny/crop512/fp16/box | 25 | hybrid | 140 | 0% | 1.11 | 86% | 88% | 89% | 9% |
| arsenal_example | tiny/crop512/fp16/box | 50 | interp | 143 | 0% | 25.96 | 22% | 41% | 47% | 31% |
| arsenal_example | tiny/crop512/fp16/box | 50 | fwd | 143 | 41% | 1.18 | 54% | 55% | 55% | 6% |
| arsenal_example | tiny/crop512/fp16/box | 50 | bwd | 143 | 6% | 1.29 | 66% | 67% | 69% | 25% |
| arsenal_example | tiny/crop512/fp16/box | 50 | nearest | 143 | 22% | 1.19 | 64% | 65% | 66% | 14% |
| arsenal_example | tiny/crop512/fp16/box | 50 | scorew | 143 | 2% | 1.13 | 83% | 85% | 86% | 10% |
| arsenal_example | tiny/crop512/fp16/box | 50 | hybrid | 143 | 0% | 1.15 | 83% | 85% | 86% | 12% |
| arsenal_example | tiny/full/fp16/box | 10 | interp | 133 | 0% | 4.72 | 52% | 69% | 87% | 3% |
| arsenal_example | tiny/full/fp16/box | 10 | fwd | 133 | 8% | 2.06 | 68% | 76% | 77% | 13% |
| arsenal_example | tiny/full/fp16/box | 10 | bwd | 133 | 0% | 2.25 | 66% | 69% | 77% | 18% |
| arsenal_example | tiny/full/fp16/box | 10 | nearest | 133 | 5% | 1.99 | 71% | 75% | 80% | 12% |
| arsenal_example | tiny/full/fp16/box | 10 | scorew | 133 | 0% | 2.1 | 72% | 77% | 81% | 14% |
| arsenal_example | tiny/full/fp16/box | 10 | hybrid | 133 | 0% | 2.1 | 72% | 77% | 81% | 14% |
| arsenal_example | tiny/full/fp16/box | 25 | interp | 140 | 0% | 9.74 | 34% | 51% | 62% | 9% |
| arsenal_example | tiny/full/fp16/box | 25 | fwd | 140 | 7% | 3.78 | 55% | 69% | 71% | 24% |
| arsenal_example | tiny/full/fp16/box | 25 | bwd | 140 | 0% | 2.91 | 71% | 80% | 82% | 15% |
| arsenal_example | tiny/full/fp16/box | 25 | nearest | 140 | 7% | 2.74 | 64% | 76% | 80% | 12% |
| arsenal_example | tiny/full/fp16/box | 25 | scorew | 140 | 0% | 2.89 | 67% | 80% | 84% | 13% |
| arsenal_example | tiny/full/fp16/box | 25 | hybrid | 140 | 0% | 2.89 | 67% | 80% | 84% | 13% |
| arsenal_example | tiny/full/fp16/box | 50 | interp | 143 | 0% | 25.96 | 22% | 41% | 47% | 31% |
| arsenal_example | tiny/full/fp16/box | 50 | fwd | 143 | 7% | 6.36 | 36% | 52% | 55% | 41% |
| arsenal_example | tiny/full/fp16/box | 50 | bwd | 143 | 0% | 4.21 | 51% | 53% | 54% | 41% |
| arsenal_example | tiny/full/fp16/box | 50 | nearest | 143 | 7% | 4.09 | 54% | 64% | 66% | 28% |
| arsenal_example | tiny/full/fp16/box | 50 | scorew | 143 | 0% | 4.21 | 55% | 64% | 66% | 32% |
| arsenal_example | tiny/full/fp16/box | 50 | hybrid | 143 | 0% | 4.21 | 55% | 64% | 66% | 32% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 10 | interp | 505 | 0% | 2.01 | 74% | 89% | 96% | 1% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 10 | fwd | 505 | 3% | 1.09 | 90% | 94% | 95% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 10 | bwd | 505 | 1% | 1.09 | 90% | 95% | 97% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 10 | nearest | 505 | 1% | 1.06 | 93% | 97% | 97% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 10 | scorew | 505 | 1% | 1.08 | 93% | 97% | 98% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 10 | hybrid | 505 | 0% | 1.08 | 93% | 98% | 98% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 25 | interp | 539 | 0% | 7.26 | 42% | 59% | 73% | 9% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 25 | fwd | 539 | 3% | 1.11 | 88% | 93% | 95% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 25 | bwd | 539 | 1% | 1.18 | 83% | 88% | 92% | 6% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 25 | nearest | 539 | 1% | 1.14 | 87% | 92% | 95% | 1% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 25 | scorew | 539 | 1% | 1.16 | 88% | 93% | 96% | 1% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 25 | hybrid | 539 | 0% | 1.16 | 88% | 93% | 96% | 1% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | interp | 550 | 0% | 17.58 | 31% | 41% | 54% | 29% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | fwd | 550 | 4% | 1.12 | 86% | 92% | 94% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | bwd | 550 | 1% | 1.18 | 83% | 88% | 91% | 7% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | nearest | 550 | 1% | 1.14 | 87% | 92% | 95% | 2% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | scorew | 550 | 1% | 1.14 | 87% | 92% | 95% | 2% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | hybrid | 550 | 0% | 1.16 | 87% | 92% | 95% | 3% |
| astonvilla_seg080 | base_plus/full/fp16/box | 10 | interp | 505 | 0% | 2.01 | 74% | 89% | 96% | 1% |
| astonvilla_seg080 | base_plus/full/fp16/box | 10 | fwd | 505 | 1% | 1.14 | 89% | 96% | 98% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 10 | bwd | 505 | 1% | 1.17 | 89% | 95% | 97% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 10 | nearest | 505 | 0% | 1.12 | 92% | 97% | 98% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 10 | scorew | 505 | 0% | 1.12 | 92% | 97% | 98% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 10 | hybrid | 505 | 0% | 1.12 | 92% | 97% | 98% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 25 | interp | 539 | 0% | 7.26 | 42% | 59% | 73% | 9% |
| astonvilla_seg080 | base_plus/full/fp16/box | 25 | fwd | 539 | 1% | 1.16 | 89% | 94% | 97% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 25 | bwd | 539 | 1% | 1.24 | 85% | 90% | 94% | 1% |
| astonvilla_seg080 | base_plus/full/fp16/box | 25 | nearest | 539 | 0% | 1.19 | 89% | 94% | 97% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 25 | scorew | 539 | 0% | 1.19 | 89% | 94% | 97% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 25 | hybrid | 539 | 0% | 1.19 | 89% | 94% | 97% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | interp | 550 | 0% | 17.58 | 31% | 41% | 54% | 29% |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | fwd | 550 | 5% | 1.13 | 86% | 91% | 94% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | bwd | 550 | 1% | 1.25 | 81% | 86% | 90% | 5% |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | nearest | 550 | 1% | 1.18 | 85% | 90% | 94% | 1% |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | scorew | 550 | 0% | 1.19 | 85% | 91% | 95% | 1% |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | hybrid | 550 | 0% | 1.19 | 85% | 91% | 95% | 1% |
| astonvilla_seg080 | small/crop512/fp16/box | 10 | interp | 505 | 0% | 2.01 | 74% | 89% | 96% | 1% |
| astonvilla_seg080 | small/crop512/fp16/box | 10 | fwd | 505 | 2% | 1.37 | 90% | 95% | 96% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 10 | bwd | 505 | 1% | 1.36 | 91% | 96% | 97% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 10 | nearest | 505 | 1% | 1.34 | 93% | 97% | 97% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 10 | scorew | 505 | 1% | 1.34 | 94% | 97% | 98% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 10 | hybrid | 505 | 0% | 1.35 | 94% | 98% | 98% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 25 | interp | 539 | 0% | 7.26 | 42% | 59% | 73% | 9% |
| astonvilla_seg080 | small/crop512/fp16/box | 25 | fwd | 539 | 1% | 1.42 | 89% | 95% | 97% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 25 | bwd | 539 | 1% | 1.44 | 86% | 91% | 95% | 2% |
| astonvilla_seg080 | small/crop512/fp16/box | 25 | nearest | 539 | 0% | 1.42 | 91% | 94% | 97% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 25 | scorew | 539 | 0% | 1.42 | 91% | 94% | 97% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 25 | hybrid | 539 | 0% | 1.42 | 91% | 94% | 97% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | interp | 550 | 0% | 17.58 | 31% | 41% | 54% | 29% |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | fwd | 550 | 2% | 1.44 | 86% | 93% | 96% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | bwd | 550 | 1% | 1.49 | 82% | 88% | 91% | 7% |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | nearest | 550 | 1% | 1.45 | 86% | 92% | 95% | 2% |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | scorew | 550 | 0% | 1.45 | 86% | 92% | 95% | 2% |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | hybrid | 550 | 0% | 1.45 | 86% | 92% | 95% | 3% |
| astonvilla_seg080 | small/full/fp16/box | 10 | interp | 505 | 0% | 2.01 | 74% | 89% | 96% | 1% |
| astonvilla_seg080 | small/full/fp16/box | 10 | fwd | 505 | 0% | 1.49 | 90% | 97% | 98% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 10 | bwd | 505 | 1% | 1.5 | 88% | 95% | 97% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 10 | nearest | 505 | 0% | 1.46 | 92% | 97% | 98% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 10 | scorew | 505 | 0% | 1.46 | 92% | 97% | 98% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 10 | hybrid | 505 | 0% | 1.46 | 92% | 97% | 98% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 25 | interp | 539 | 0% | 7.26 | 42% | 59% | 73% | 9% |
| astonvilla_seg080 | small/full/fp16/box | 25 | fwd | 539 | 1% | 1.5 | 90% | 96% | 98% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 25 | bwd | 539 | 1% | 1.53 | 84% | 90% | 94% | 2% |
| astonvilla_seg080 | small/full/fp16/box | 25 | nearest | 539 | 0% | 1.52 | 88% | 94% | 96% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 25 | scorew | 539 | 0% | 1.52 | 88% | 94% | 96% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 25 | hybrid | 539 | 0% | 1.52 | 88% | 94% | 96% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 50 | interp | 550 | 0% | 17.58 | 31% | 41% | 54% | 29% |
| astonvilla_seg080 | small/full/fp16/box | 50 | fwd | 550 | 1% | 1.51 | 89% | 96% | 98% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 50 | bwd | 550 | 1% | 1.57 | 80% | 86% | 90% | 7% |
| astonvilla_seg080 | small/full/fp16/box | 50 | nearest | 550 | 1% | 1.56 | 85% | 91% | 94% | 2% |
| astonvilla_seg080 | small/full/fp16/box | 50 | scorew | 550 | 0% | 1.56 | 85% | 91% | 95% | 2% |
| astonvilla_seg080 | small/full/fp16/box | 50 | hybrid | 550 | 0% | 1.56 | 85% | 91% | 95% | 2% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 10 | interp | 505 | 0% | 2.01 | 74% | 89% | 96% | 1% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 10 | fwd | 505 | 3% | 1.22 | 90% | 95% | 95% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 10 | bwd | 505 | 0% | 1.23 | 90% | 97% | 98% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 10 | nearest | 505 | 1% | 1.21 | 93% | 97% | 97% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 10 | scorew | 505 | 0% | 1.22 | 94% | 98% | 98% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 10 | hybrid | 505 | 0% | 1.22 | 94% | 98% | 98% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 25 | interp | 539 | 0% | 7.26 | 42% | 59% | 73% | 9% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 25 | fwd | 539 | 2% | 1.23 | 90% | 94% | 96% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 25 | bwd | 539 | 0% | 1.28 | 84% | 90% | 93% | 4% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 25 | nearest | 539 | 1% | 1.25 | 90% | 94% | 96% | 1% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 25 | scorew | 539 | 0% | 1.26 | 90% | 94% | 96% | 1% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 25 | hybrid | 539 | 0% | 1.26 | 90% | 94% | 96% | 1% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | interp | 550 | 0% | 17.58 | 31% | 41% | 54% | 29% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | fwd | 550 | 2% | 1.26 | 88% | 94% | 96% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | bwd | 550 | 0% | 1.29 | 81% | 88% | 90% | 7% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | nearest | 550 | 0% | 1.28 | 86% | 92% | 95% | 2% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | scorew | 550 | 0% | 1.28 | 85% | 92% | 95% | 2% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | hybrid | 550 | 0% | 1.29 | 85% | 92% | 95% | 3% |
| astonvilla_seg080 | tiny/full/fp16/box | 10 | interp | 505 | 0% | 2.01 | 74% | 89% | 96% | 1% |
| astonvilla_seg080 | tiny/full/fp16/box | 10 | fwd | 505 | 0% | 1.32 | 90% | 96% | 98% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 10 | bwd | 505 | 0% | 1.34 | 89% | 96% | 98% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 10 | nearest | 505 | 0% | 1.29 | 93% | 97% | 98% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 10 | scorew | 505 | 0% | 1.3 | 93% | 97% | 98% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 10 | hybrid | 505 | 0% | 1.3 | 93% | 97% | 98% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 25 | interp | 539 | 0% | 7.26 | 42% | 59% | 73% | 9% |
| astonvilla_seg080 | tiny/full/fp16/box | 25 | fwd | 539 | 0% | 1.34 | 89% | 96% | 98% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 25 | bwd | 539 | 0% | 1.33 | 87% | 93% | 97% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 25 | nearest | 539 | 0% | 1.32 | 89% | 95% | 98% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 25 | scorew | 539 | 0% | 1.32 | 89% | 95% | 98% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 25 | hybrid | 539 | 0% | 1.32 | 89% | 95% | 98% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | interp | 550 | 0% | 17.58 | 31% | 41% | 54% | 29% |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | fwd | 550 | 0% | 1.36 | 89% | 95% | 98% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | bwd | 550 | 0% | 1.41 | 79% | 85% | 90% | 3% |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | nearest | 550 | 0% | 1.36 | 88% | 93% | 97% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | scorew | 550 | 0% | 1.36 | 88% | 93% | 97% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | hybrid | 550 | 0% | 1.36 | 88% | 93% | 97% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 10 | interp | 195 | 0% | 1.68 | 91% | 95% | 97% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 10 | fwd | 195 | 0% | 1.05 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 10 | bwd | 195 | 0% | 1.12 | 92% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 10 | nearest | 195 | 0% | 1.05 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 10 | scorew | 195 | 0% | 1.05 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 10 | hybrid | 195 | 0% | 1.05 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 25 | interp | 208 | 0% | 6.64 | 42% | 67% | 91% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 25 | fwd | 208 | 0% | 1.04 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 25 | bwd | 208 | 0% | 1.07 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 25 | nearest | 208 | 0% | 1.06 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 25 | scorew | 208 | 0% | 1.06 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 25 | hybrid | 208 | 0% | 1.06 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 50 | interp | 212 | 0% | 32.01 | 11% | 18% | 32% | 19% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 50 | fwd | 212 | 0% | 1.06 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 50 | bwd | 212 | 0% | 1.11 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 50 | nearest | 212 | 0% | 1.06 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 50 | scorew | 212 | 0% | 1.06 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 50 | hybrid | 212 | 0% | 1.06 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 10 | interp | 195 | 0% | 1.68 | 91% | 95% | 97% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 10 | fwd | 195 | 0% | 1.18 | 92% | 98% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 10 | bwd | 195 | 0% | 1.2 | 93% | 99% | 99% | 1% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 10 | nearest | 195 | 0% | 1.21 | 94% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 10 | scorew | 195 | 0% | 1.21 | 93% | 99% | 99% | 1% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 10 | hybrid | 195 | 0% | 1.21 | 93% | 99% | 99% | 1% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 25 | interp | 208 | 0% | 6.64 | 42% | 67% | 91% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 25 | fwd | 208 | 0% | 1.17 | 94% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 25 | bwd | 208 | 0% | 1.23 | 92% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 25 | nearest | 208 | 0% | 1.19 | 94% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 25 | scorew | 208 | 0% | 1.19 | 94% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 25 | hybrid | 208 | 0% | 1.19 | 94% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 50 | interp | 212 | 0% | 32.01 | 11% | 18% | 32% | 19% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 50 | fwd | 212 | 0% | 1.21 | 92% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 50 | bwd | 212 | 0% | 1.2 | 92% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 50 | nearest | 212 | 0% | 1.23 | 94% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 50 | scorew | 212 | 0% | 1.23 | 92% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 50 | hybrid | 212 | 0% | 1.23 | 92% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 10 | interp | 195 | 0% | 1.68 | 91% | 95% | 97% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 10 | fwd | 195 | 0% | 1.21 | 92% | 98% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 10 | bwd | 195 | 0% | 1.19 | 88% | 98% | 99% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 10 | nearest | 195 | 0% | 1.19 | 90% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 10 | scorew | 195 | 0% | 1.18 | 89% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 10 | hybrid | 195 | 0% | 1.18 | 89% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 25 | interp | 208 | 0% | 6.64 | 42% | 67% | 91% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 25 | fwd | 208 | 0% | 1.11 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 25 | bwd | 208 | 0% | 1.08 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 25 | nearest | 208 | 0% | 1.1 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 25 | scorew | 208 | 0% | 1.1 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 25 | hybrid | 208 | 0% | 1.1 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 50 | interp | 212 | 0% | 32.01 | 11% | 18% | 32% | 19% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 50 | fwd | 212 | 0% | 1.13 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 50 | bwd | 212 | 0% | 1.09 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 50 | nearest | 212 | 0% | 1.08 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 50 | scorew | 212 | 0% | 1.08 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 50 | hybrid | 212 | 0% | 1.08 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 10 | interp | 195 | 0% | 1.68 | 91% | 95% | 97% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 10 | fwd | 195 | 0% | 1.16 | 96% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 10 | bwd | 195 | 0% | 1.25 | 97% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 10 | nearest | 195 | 0% | 1.15 | 98% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 10 | scorew | 195 | 0% | 1.15 | 98% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 10 | hybrid | 195 | 0% | 1.15 | 98% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 25 | interp | 208 | 0% | 6.64 | 42% | 67% | 91% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 25 | fwd | 208 | 0% | 1.13 | 96% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 25 | bwd | 208 | 0% | 1.13 | 97% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 25 | nearest | 208 | 0% | 1.15 | 96% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 25 | scorew | 208 | 0% | 1.15 | 96% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 25 | hybrid | 208 | 0% | 1.15 | 96% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 50 | interp | 212 | 0% | 32.01 | 11% | 18% | 32% | 19% |
| bournemouth_1st_seg070 | small/full/fp16/box | 50 | fwd | 212 | 0% | 1.18 | 97% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 50 | bwd | 212 | 0% | 1.14 | 97% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 50 | nearest | 212 | 0% | 1.14 | 97% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 50 | scorew | 212 | 0% | 1.14 | 97% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 50 | hybrid | 212 | 0% | 1.14 | 97% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 10 | interp | 195 | 0% | 1.68 | 91% | 95% | 97% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 10 | fwd | 195 | 0% | 1.03 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 10 | bwd | 195 | 0% | 1.06 | 92% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 10 | nearest | 195 | 0% | 1.03 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 10 | scorew | 195 | 0% | 1.03 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 10 | hybrid | 195 | 0% | 1.03 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 25 | interp | 208 | 0% | 6.64 | 42% | 67% | 91% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 25 | fwd | 208 | 0% | 0.97 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 25 | bwd | 208 | 0% | 1.07 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 25 | nearest | 208 | 0% | 1.04 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 25 | scorew | 208 | 0% | 1.04 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 25 | hybrid | 208 | 0% | 1.04 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 50 | interp | 212 | 0% | 32.01 | 11% | 18% | 32% | 19% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 50 | fwd | 212 | 0% | 1.03 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 50 | bwd | 212 | 0% | 1.06 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 50 | nearest | 212 | 0% | 1.06 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 50 | scorew | 212 | 0% | 1.04 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 50 | hybrid | 212 | 0% | 1.04 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 10 | interp | 195 | 0% | 1.68 | 91% | 95% | 97% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 10 | fwd | 195 | 0% | 1.0 | 93% | 97% | 99% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 10 | bwd | 195 | 0% | 1.03 | 95% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 10 | nearest | 195 | 0% | 1.03 | 95% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 10 | scorew | 195 | 0% | 1.03 | 96% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 10 | hybrid | 195 | 0% | 1.03 | 96% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 25 | interp | 208 | 0% | 6.64 | 42% | 67% | 91% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 25 | fwd | 208 | 0% | 0.96 | 95% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 25 | bwd | 208 | 0% | 0.92 | 97% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 25 | nearest | 208 | 0% | 0.93 | 95% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 25 | scorew | 208 | 0% | 0.93 | 95% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 25 | hybrid | 208 | 0% | 0.93 | 95% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 50 | interp | 212 | 0% | 32.01 | 11% | 18% | 32% | 19% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 50 | fwd | 212 | 0% | 1.04 | 96% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 50 | bwd | 212 | 0% | 0.99 | 97% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 50 | nearest | 212 | 0% | 0.99 | 97% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 50 | scorew | 212 | 0% | 0.99 | 98% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 50 | hybrid | 212 | 0% | 0.99 | 98% | 100% | 100% | 0% |

### Forward/backward agreement (both tracks present and within 5 px): coverage and precision

| clip | config | N | held-out | both present | agree <=5px (coverage) | hit@10 of agreeing | hit@20 of agreeing | median err |
|---|---|---|---|---|---|---|---|---|
| arsenal_example | base_plus/crop512/fp16/box | 10 | 133 | 112 | 100 (75%) | 100% | 100% | 0.93 |
| arsenal_example | base_plus/crop512/fp16/box | 25 | 140 | 113 | 109 (78%) | 100% | 100% | 1.02 |
| arsenal_example | base_plus/crop512/fp16/box | 50 | 143 | 90 | 88 (62%) | 100% | 100% | 0.95 |
| arsenal_example | base_plus/full/fp16/box | 10 | 133 | 124 | 94 (71%) | 100% | 100% | 1.31 |
| arsenal_example | base_plus/full/fp16/box | 25 | 140 | 129 | 94 (67%) | 99% | 100% | 1.52 |
| arsenal_example | base_plus/full/fp16/box | 50 | 143 | 132 | 50 (35%) | 100% | 100% | 1.25 |
| arsenal_example | small/crop512/fp16/box | 10 | 133 | 115 | 102 (77%) | 100% | 100% | 0.99 |
| arsenal_example | small/crop512/fp16/box | 25 | 140 | 116 | 113 (81%) | 100% | 100% | 1.09 |
| arsenal_example | small/crop512/fp16/box | 50 | 143 | 94 | 92 (64%) | 100% | 100% | 0.94 |
| arsenal_example | small/full/fp16/box | 10 | 133 | 119 | 83 (62%) | 100% | 100% | 1.42 |
| arsenal_example | small/full/fp16/box | 25 | 140 | 116 | 91 (65%) | 100% | 100% | 1.91 |
| arsenal_example | small/full/fp16/box | 50 | 143 | 116 | 69 (48%) | 100% | 100% | 1.65 |
| arsenal_example | tiny/crop512/fp16/box | 10 | 133 | 117 | 99 (74%) | 100% | 100% | 1.0 |
| arsenal_example | tiny/crop512/fp16/box | 25 | 140 | 104 | 95 (68%) | 100% | 100% | 1.04 |
| arsenal_example | tiny/crop512/fp16/box | 50 | 143 | 78 | 50 (35%) | 100% | 100% | 0.96 |
| arsenal_example | tiny/full/fp16/box | 10 | 133 | 123 | 84 (63%) | 100% | 100% | 1.3 |
| arsenal_example | tiny/full/fp16/box | 25 | 140 | 130 | 89 (64%) | 99% | 100% | 2.12 |
| arsenal_example | tiny/full/fp16/box | 50 | 143 | 133 | 45 (31%) | 100% | 100% | 2.37 |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 10 | 505 | 489 | 434 (86%) | 99% | 99% | 1.02 |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 25 | 539 | 525 | 461 (86%) | 96% | 99% | 1.07 |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | 550 | 529 | 473 (86%) | 96% | 99% | 1.06 |
| astonvilla_seg080 | base_plus/full/fp16/box | 10 | 505 | 498 | 438 (87%) | 98% | 99% | 1.06 |
| astonvilla_seg080 | base_plus/full/fp16/box | 25 | 539 | 532 | 471 (87%) | 96% | 99% | 1.09 |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | 550 | 523 | 444 (81%) | 98% | 99% | 1.06 |
| astonvilla_seg080 | small/crop512/fp16/box | 10 | 505 | 493 | 432 (86%) | 99% | 99% | 1.3 |
| astonvilla_seg080 | small/crop512/fp16/box | 25 | 539 | 531 | 483 (90%) | 96% | 99% | 1.36 |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | 550 | 538 | 472 (86%) | 96% | 99% | 1.35 |
| astonvilla_seg080 | small/full/fp16/box | 10 | 505 | 501 | 431 (85%) | 99% | 99% | 1.39 |
| astonvilla_seg080 | small/full/fp16/box | 25 | 539 | 535 | 458 (85%) | 97% | 99% | 1.4 |
| astonvilla_seg080 | small/full/fp16/box | 50 | 550 | 544 | 450 (82%) | 98% | 99% | 1.44 |
| astonvilla_seg080 | tiny/crop512/fp16/box | 10 | 505 | 491 | 436 (86%) | 99% | 99% | 1.14 |
| astonvilla_seg080 | tiny/crop512/fp16/box | 25 | 539 | 529 | 467 (87%) | 97% | 99% | 1.2 |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | 550 | 539 | 471 (86%) | 97% | 99% | 1.21 |
| astonvilla_seg080 | tiny/full/fp16/box | 10 | 505 | 505 | 439 (87%) | 99% | 99% | 1.21 |
| astonvilla_seg080 | tiny/full/fp16/box | 25 | 539 | 538 | 481 (89%) | 97% | 99% | 1.22 |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | 550 | 548 | 434 (79%) | 98% | 99% | 1.21 |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 10 | 195 | 195 | 190 (97%) | 100% | 100% | 1.04 |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 25 | 208 | 208 | 208 (100%) | 100% | 100% | 1.06 |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 50 | 212 | 212 | 212 (100%) | 100% | 100% | 1.06 |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 10 | 195 | 195 | 187 (96%) | 100% | 100% | 1.13 |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 25 | 208 | 208 | 207 (100%) | 100% | 100% | 1.17 |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 50 | 212 | 212 | 209 (99%) | 100% | 100% | 1.21 |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 10 | 195 | 195 | 169 (87%) | 100% | 100% | 0.94 |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 25 | 208 | 208 | 208 (100%) | 100% | 100% | 1.1 |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 50 | 212 | 212 | 212 (100%) | 100% | 100% | 1.08 |
| bournemouth_1st_seg070 | small/full/fp16/box | 10 | 195 | 195 | 187 (96%) | 100% | 100% | 1.12 |
| bournemouth_1st_seg070 | small/full/fp16/box | 25 | 208 | 208 | 205 (99%) | 100% | 100% | 1.14 |
| bournemouth_1st_seg070 | small/full/fp16/box | 50 | 212 | 212 | 209 (99%) | 100% | 100% | 1.14 |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 10 | 195 | 195 | 188 (96%) | 100% | 100% | 1.01 |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 25 | 208 | 208 | 208 (100%) | 100% | 100% | 1.04 |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 50 | 212 | 212 | 212 (100%) | 100% | 100% | 1.04 |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 10 | 195 | 195 | 180 (92%) | 100% | 100% | 0.96 |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 25 | 208 | 208 | 206 (99%) | 100% | 100% | 0.93 |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 50 | 212 | 212 | 209 (99%) | 100% | 100% | 0.99 |

### Carry curve: single prompt, error vs distance from the anchor frame

(pooled over all anchors of all N; frames that are anchors in any N excluded; `hold` = baseline that keeps the anchor position)

| clip | config | direction | distance (frames) | n held-out | SAM2 absent | median err px | hit@5 | hit@10 | hit@20 | wrong-object (>50px, of present) |
|---|---|---|---|---|---|---|---|---|---|---|
| arsenal_example | base_plus/crop512/fp16/box | fwd | 1-5 | 88 | 12% | 1.09 | 81% | 81% | 81% | 6% |
| arsenal_example | base_plus/crop512/fp16/box | fwd | 6-12 | 88 | 6% | 1.18 | 80% | 80% | 80% | 14% |
| arsenal_example | base_plus/crop512/fp16/box | fwd | 13-25 | 98 | 5% | 1.15 | 87% | 87% | 87% | 8% |
| arsenal_example | base_plus/crop512/fp16/box | fwd | 26-50 | 37 | 22% | 1.18 | 76% | 76% | 76% | 0% |
| arsenal_example | base_plus/crop512/fp16/box | bwd | 1-5 | 88 | 5% | 1.02 | 83% | 84% | 85% | 7% |
| arsenal_example | base_plus/crop512/fp16/box | bwd | 6-12 | 102 | 6% | 1.02 | 83% | 84% | 85% | 9% |
| arsenal_example | base_plus/crop512/fp16/box | bwd | 13-25 | 125 | 10% | 0.88 | 76% | 78% | 79% | 12% |
| arsenal_example | base_plus/crop512/fp16/box | bwd | 26-50 | 37 | 32% | 1.09 | 57% | 62% | 65% | 4% |
| arsenal_example | base_plus/crop512/fp16/box | hold-anchor | 1-5 | 88 | - | 10.62 | 24% | 47% | 72% | - |
| arsenal_example | base_plus/crop512/fp16/box | hold-anchor | 6-12 | 88 | - | 28.54 | 3% | 15% | 34% | - |
| arsenal_example | base_plus/crop512/fp16/box | hold-anchor | 13-25 | 98 | - | 60.41 | 0% | 3% | 10% | - |
| arsenal_example | base_plus/crop512/fp16/box | hold-anchor | 26-50 | 37 | - | 131.76 | 0% | 0% | 5% | - |
| arsenal_example | base_plus/full/fp16/box | fwd | 1-5 | 88 | 6% | 1.62 | 76% | 77% | 78% | 10% |
| arsenal_example | base_plus/full/fp16/box | fwd | 6-12 | 88 | 6% | 1.73 | 72% | 75% | 76% | 19% |
| arsenal_example | base_plus/full/fp16/box | fwd | 13-25 | 98 | 0% | 1.62 | 66% | 72% | 73% | 24% |
| arsenal_example | base_plus/full/fp16/box | fwd | 26-50 | 37 | 0% | 509.13 | 14% | 16% | 19% | 81% |
| arsenal_example | base_plus/full/fp16/box | bwd | 1-5 | 88 | 0% | 2.02 | 76% | 82% | 85% | 11% |
| arsenal_example | base_plus/full/fp16/box | bwd | 6-12 | 102 | 0% | 1.82 | 78% | 78% | 82% | 18% |
| arsenal_example | base_plus/full/fp16/box | bwd | 13-25 | 125 | 0% | 2.07 | 66% | 70% | 73% | 25% |
| arsenal_example | base_plus/full/fp16/box | bwd | 26-50 | 37 | 0% | 221.77 | 32% | 38% | 41% | 59% |
| arsenal_example | base_plus/full/fp16/box | hold-anchor | 1-5 | 88 | - | 10.62 | 24% | 47% | 72% | - |
| arsenal_example | base_plus/full/fp16/box | hold-anchor | 6-12 | 88 | - | 28.54 | 3% | 15% | 34% | - |
| arsenal_example | base_plus/full/fp16/box | hold-anchor | 13-25 | 98 | - | 60.41 | 0% | 3% | 10% | - |
| arsenal_example | base_plus/full/fp16/box | hold-anchor | 26-50 | 37 | - | 131.76 | 0% | 0% | 5% | - |
| arsenal_example | small/crop512/fp16/box | fwd | 1-5 | 88 | 9% | 1.14 | 82% | 82% | 82% | 9% |
| arsenal_example | small/crop512/fp16/box | fwd | 6-12 | 88 | 11% | 1.18 | 82% | 82% | 82% | 8% |
| arsenal_example | small/crop512/fp16/box | fwd | 13-25 | 98 | 8% | 1.05 | 87% | 87% | 87% | 4% |
| arsenal_example | small/crop512/fp16/box | fwd | 26-50 | 37 | 16% | 1.34 | 76% | 76% | 76% | 6% |
| arsenal_example | small/crop512/fp16/box | bwd | 1-5 | 88 | 5% | 1.23 | 83% | 84% | 85% | 7% |
| arsenal_example | small/crop512/fp16/box | bwd | 6-12 | 102 | 6% | 1.15 | 83% | 84% | 85% | 9% |
| arsenal_example | small/crop512/fp16/box | bwd | 13-25 | 125 | 9% | 1.23 | 77% | 78% | 80% | 12% |
| arsenal_example | small/crop512/fp16/box | bwd | 26-50 | 37 | 32% | 1.22 | 59% | 65% | 65% | 4% |
| arsenal_example | small/crop512/fp16/box | hold-anchor | 1-5 | 88 | - | 10.62 | 24% | 47% | 72% | - |
| arsenal_example | small/crop512/fp16/box | hold-anchor | 6-12 | 88 | - | 28.54 | 3% | 15% | 34% | - |
| arsenal_example | small/crop512/fp16/box | hold-anchor | 13-25 | 98 | - | 60.41 | 0% | 3% | 10% | - |
| arsenal_example | small/crop512/fp16/box | hold-anchor | 26-50 | 37 | - | 131.76 | 0% | 0% | 5% | - |
| arsenal_example | small/full/fp16/box | fwd | 1-5 | 88 | 9% | 1.9 | 77% | 77% | 78% | 12% |
| arsenal_example | small/full/fp16/box | fwd | 6-12 | 88 | 9% | 1.82 | 62% | 73% | 73% | 20% |
| arsenal_example | small/full/fp16/box | fwd | 13-25 | 98 | 13% | 1.68 | 67% | 72% | 73% | 15% |
| arsenal_example | small/full/fp16/box | fwd | 26-50 | 37 | 5% | 1.46 | 65% | 65% | 65% | 29% |
| arsenal_example | small/full/fp16/box | bwd | 1-5 | 88 | 1% | 1.94 | 69% | 75% | 82% | 11% |
| arsenal_example | small/full/fp16/box | bwd | 6-12 | 102 | 0% | 1.93 | 70% | 78% | 81% | 18% |
| arsenal_example | small/full/fp16/box | bwd | 13-25 | 125 | 0% | 4.77 | 52% | 68% | 70% | 26% |
| arsenal_example | small/full/fp16/box | bwd | 26-50 | 37 | 0% | 135.09 | 14% | 19% | 22% | 78% |
| arsenal_example | small/full/fp16/box | hold-anchor | 1-5 | 88 | - | 10.62 | 24% | 47% | 72% | - |
| arsenal_example | small/full/fp16/box | hold-anchor | 6-12 | 88 | - | 28.54 | 3% | 15% | 34% | - |
| arsenal_example | small/full/fp16/box | hold-anchor | 13-25 | 98 | - | 60.41 | 0% | 3% | 10% | - |
| arsenal_example | small/full/fp16/box | hold-anchor | 26-50 | 37 | - | 131.76 | 0% | 0% | 5% | - |
| arsenal_example | tiny/crop512/fp16/box | fwd | 1-5 | 88 | 9% | 1.09 | 81% | 81% | 81% | 9% |
| arsenal_example | tiny/crop512/fp16/box | fwd | 6-12 | 88 | 15% | 1.18 | 76% | 76% | 76% | 9% |
| arsenal_example | tiny/crop512/fp16/box | fwd | 13-25 | 98 | 15% | 1.22 | 76% | 76% | 76% | 10% |
| arsenal_example | tiny/crop512/fp16/box | fwd | 26-50 | 37 | 73% | 2.11 | 19% | 19% | 19% | 20% |
| arsenal_example | tiny/crop512/fp16/box | bwd | 1-5 | 88 | 2% | 1.14 | 84% | 84% | 85% | 9% |
| arsenal_example | tiny/crop512/fp16/box | bwd | 6-12 | 102 | 2% | 1.17 | 82% | 84% | 85% | 13% |
| arsenal_example | tiny/crop512/fp16/box | bwd | 13-25 | 125 | 2% | 1.24 | 75% | 76% | 77% | 20% |
| arsenal_example | tiny/crop512/fp16/box | bwd | 26-50 | 37 | 5% | 1.36 | 59% | 62% | 65% | 31% |
| arsenal_example | tiny/crop512/fp16/box | hold-anchor | 1-5 | 88 | - | 10.62 | 24% | 47% | 72% | - |
| arsenal_example | tiny/crop512/fp16/box | hold-anchor | 6-12 | 88 | - | 28.54 | 3% | 15% | 34% | - |
| arsenal_example | tiny/crop512/fp16/box | hold-anchor | 13-25 | 98 | - | 60.41 | 0% | 3% | 10% | - |
| arsenal_example | tiny/crop512/fp16/box | hold-anchor | 26-50 | 37 | - | 131.76 | 0% | 0% | 5% | - |
| arsenal_example | tiny/full/fp16/box | fwd | 1-5 | 88 | 7% | 1.98 | 72% | 77% | 78% | 12% |
| arsenal_example | tiny/full/fp16/box | fwd | 6-12 | 88 | 6% | 2.2 | 61% | 72% | 74% | 20% |
| arsenal_example | tiny/full/fp16/box | fwd | 13-25 | 98 | 2% | 2.79 | 60% | 73% | 73% | 25% |
| arsenal_example | tiny/full/fp16/box | fwd | 26-50 | 37 | 0% | 263.25 | 8% | 16% | 19% | 81% |
| arsenal_example | tiny/full/fp16/box | bwd | 1-5 | 88 | 0% | 2.48 | 69% | 75% | 84% | 11% |
| arsenal_example | tiny/full/fp16/box | bwd | 6-12 | 102 | 0% | 2.23 | 65% | 74% | 77% | 19% |
| arsenal_example | tiny/full/fp16/box | bwd | 13-25 | 125 | 0% | 4.2 | 57% | 62% | 63% | 34% |
| arsenal_example | tiny/full/fp16/box | bwd | 26-50 | 37 | 0% | 134.5 | 11% | 14% | 14% | 76% |
| arsenal_example | tiny/full/fp16/box | hold-anchor | 1-5 | 88 | - | 10.62 | 24% | 47% | 72% | - |
| arsenal_example | tiny/full/fp16/box | hold-anchor | 6-12 | 88 | - | 28.54 | 3% | 15% | 34% | - |
| arsenal_example | tiny/full/fp16/box | hold-anchor | 13-25 | 98 | - | 60.41 | 0% | 3% | 10% | - |
| arsenal_example | tiny/full/fp16/box | hold-anchor | 26-50 | 37 | - | 131.76 | 0% | 0% | 5% | - |
| astonvilla_seg080 | base_plus/crop512/fp16/box | fwd | 1-5 | 316 | 2% | 1.12 | 92% | 95% | 96% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | fwd | 6-12 | 406 | 5% | 1.05 | 87% | 92% | 93% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | fwd | 13-25 | 696 | 4% | 1.19 | 81% | 89% | 92% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | fwd | 26-50 | 234 | 7% | 1.16 | 83% | 91% | 92% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | bwd | 1-5 | 325 | 0% | 1.1 | 93% | 98% | 99% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | bwd | 6-12 | 424 | 1% | 1.17 | 83% | 88% | 93% | 2% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | bwd | 13-25 | 738 | 1% | 1.17 | 80% | 84% | 88% | 7% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | bwd | 26-50 | 300 | 0% | 1.13 | 81% | 84% | 88% | 10% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | hold-anchor | 1-5 | 316 | - | 5.12 | 49% | 64% | 81% | - |
| astonvilla_seg080 | base_plus/crop512/fp16/box | hold-anchor | 6-12 | 406 | - | 19.36 | 22% | 40% | 51% | - |
| astonvilla_seg080 | base_plus/crop512/fp16/box | hold-anchor | 13-25 | 696 | - | 37.53 | 10% | 21% | 37% | - |
| astonvilla_seg080 | base_plus/crop512/fp16/box | hold-anchor | 26-50 | 234 | - | 104.75 | 9% | 9% | 12% | - |
| astonvilla_seg080 | base_plus/full/fp16/box | fwd | 1-5 | 316 | 0% | 1.18 | 92% | 97% | 98% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | fwd | 6-12 | 406 | 2% | 1.11 | 87% | 94% | 97% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | fwd | 13-25 | 696 | 2% | 1.23 | 83% | 90% | 94% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | fwd | 26-50 | 234 | 10% | 1.16 | 82% | 88% | 90% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | bwd | 1-5 | 325 | 0% | 1.24 | 91% | 97% | 98% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | bwd | 6-12 | 424 | 1% | 1.25 | 84% | 90% | 95% | 1% |
| astonvilla_seg080 | base_plus/full/fp16/box | bwd | 13-25 | 738 | 1% | 1.22 | 80% | 85% | 88% | 6% |
| astonvilla_seg080 | base_plus/full/fp16/box | bwd | 26-50 | 300 | 0% | 1.29 | 73% | 78% | 81% | 16% |
| astonvilla_seg080 | base_plus/full/fp16/box | hold-anchor | 1-5 | 316 | - | 5.12 | 49% | 64% | 81% | - |
| astonvilla_seg080 | base_plus/full/fp16/box | hold-anchor | 6-12 | 406 | - | 19.36 | 22% | 40% | 51% | - |
| astonvilla_seg080 | base_plus/full/fp16/box | hold-anchor | 13-25 | 696 | - | 37.53 | 10% | 21% | 37% | - |
| astonvilla_seg080 | base_plus/full/fp16/box | hold-anchor | 26-50 | 234 | - | 104.75 | 9% | 9% | 12% | - |
| astonvilla_seg080 | small/crop512/fp16/box | fwd | 1-5 | 316 | 2% | 1.35 | 93% | 96% | 97% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | fwd | 6-12 | 406 | 3% | 1.39 | 89% | 94% | 96% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | fwd | 13-25 | 696 | 3% | 1.49 | 81% | 90% | 93% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | fwd | 26-50 | 234 | 5% | 1.51 | 82% | 93% | 95% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | bwd | 1-5 | 325 | 0% | 1.38 | 94% | 98% | 99% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | bwd | 6-12 | 424 | 1% | 1.44 | 85% | 92% | 96% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | bwd | 13-25 | 738 | 1% | 1.44 | 79% | 85% | 89% | 6% |
| astonvilla_seg080 | small/crop512/fp16/box | bwd | 26-50 | 300 | 0% | 1.53 | 73% | 77% | 81% | 10% |
| astonvilla_seg080 | small/crop512/fp16/box | hold-anchor | 1-5 | 316 | - | 5.12 | 49% | 64% | 81% | - |
| astonvilla_seg080 | small/crop512/fp16/box | hold-anchor | 6-12 | 406 | - | 19.36 | 22% | 40% | 51% | - |
| astonvilla_seg080 | small/crop512/fp16/box | hold-anchor | 13-25 | 696 | - | 37.53 | 10% | 21% | 37% | - |
| astonvilla_seg080 | small/crop512/fp16/box | hold-anchor | 26-50 | 234 | - | 104.75 | 9% | 9% | 12% | - |
| astonvilla_seg080 | small/full/fp16/box | fwd | 1-5 | 316 | 0% | 1.46 | 93% | 97% | 98% | 0% |
| astonvilla_seg080 | small/full/fp16/box | fwd | 6-12 | 406 | 1% | 1.49 | 87% | 96% | 98% | 0% |
| astonvilla_seg080 | small/full/fp16/box | fwd | 13-25 | 696 | 1% | 1.56 | 84% | 92% | 94% | 0% |
| astonvilla_seg080 | small/full/fp16/box | fwd | 26-50 | 234 | 2% | 1.49 | 87% | 96% | 98% | 0% |
| astonvilla_seg080 | small/full/fp16/box | bwd | 1-5 | 325 | 0% | 1.53 | 90% | 96% | 97% | 0% |
| astonvilla_seg080 | small/full/fp16/box | bwd | 6-12 | 424 | 1% | 1.53 | 83% | 90% | 96% | 0% |
| astonvilla_seg080 | small/full/fp16/box | bwd | 13-25 | 738 | 1% | 1.51 | 77% | 83% | 88% | 6% |
| astonvilla_seg080 | small/full/fp16/box | bwd | 26-50 | 300 | 0% | 1.65 | 70% | 73% | 76% | 15% |
| astonvilla_seg080 | small/full/fp16/box | hold-anchor | 1-5 | 316 | - | 5.12 | 49% | 64% | 81% | - |
| astonvilla_seg080 | small/full/fp16/box | hold-anchor | 6-12 | 406 | - | 19.36 | 22% | 40% | 51% | - |
| astonvilla_seg080 | small/full/fp16/box | hold-anchor | 13-25 | 696 | - | 37.53 | 10% | 21% | 37% | - |
| astonvilla_seg080 | small/full/fp16/box | hold-anchor | 26-50 | 234 | - | 104.75 | 9% | 9% | 12% | - |
| astonvilla_seg080 | tiny/crop512/fp16/box | fwd | 1-5 | 316 | 2% | 1.23 | 93% | 96% | 96% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | fwd | 6-12 | 406 | 4% | 1.2 | 89% | 93% | 94% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | fwd | 13-25 | 696 | 3% | 1.31 | 83% | 89% | 92% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | fwd | 26-50 | 234 | 5% | 1.31 | 87% | 94% | 95% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | bwd | 1-5 | 325 | 0% | 1.23 | 94% | 98% | 99% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | bwd | 6-12 | 424 | 0% | 1.29 | 83% | 91% | 94% | 1% |
| astonvilla_seg080 | tiny/crop512/fp16/box | bwd | 13-25 | 738 | 0% | 1.28 | 78% | 84% | 87% | 7% |
| astonvilla_seg080 | tiny/crop512/fp16/box | bwd | 26-50 | 300 | 0% | 1.35 | 71% | 75% | 78% | 14% |
| astonvilla_seg080 | tiny/crop512/fp16/box | hold-anchor | 1-5 | 316 | - | 5.12 | 49% | 64% | 81% | - |
| astonvilla_seg080 | tiny/crop512/fp16/box | hold-anchor | 6-12 | 406 | - | 19.36 | 22% | 40% | 51% | - |
| astonvilla_seg080 | tiny/crop512/fp16/box | hold-anchor | 13-25 | 696 | - | 37.53 | 10% | 21% | 37% | - |
| astonvilla_seg080 | tiny/crop512/fp16/box | hold-anchor | 26-50 | 234 | - | 104.75 | 9% | 9% | 12% | - |
| astonvilla_seg080 | tiny/full/fp16/box | fwd | 1-5 | 316 | 0% | 1.28 | 95% | 97% | 98% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | fwd | 6-12 | 406 | 0% | 1.34 | 87% | 95% | 98% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | fwd | 13-25 | 696 | 0% | 1.42 | 84% | 92% | 95% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | fwd | 26-50 | 234 | 1% | 1.38 | 87% | 96% | 97% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | bwd | 1-5 | 325 | 0% | 1.34 | 90% | 97% | 97% | 1% |
| astonvilla_seg080 | tiny/full/fp16/box | bwd | 6-12 | 424 | 0% | 1.43 | 85% | 92% | 96% | 2% |
| astonvilla_seg080 | tiny/full/fp16/box | bwd | 13-25 | 738 | 0% | 1.35 | 79% | 86% | 89% | 6% |
| astonvilla_seg080 | tiny/full/fp16/box | bwd | 26-50 | 300 | 0% | 1.62 | 65% | 68% | 71% | 20% |
| astonvilla_seg080 | tiny/full/fp16/box | hold-anchor | 1-5 | 316 | - | 5.12 | 49% | 64% | 81% | - |
| astonvilla_seg080 | tiny/full/fp16/box | hold-anchor | 6-12 | 406 | - | 19.36 | 22% | 40% | 51% | - |
| astonvilla_seg080 | tiny/full/fp16/box | hold-anchor | 13-25 | 696 | - | 37.53 | 10% | 21% | 37% | - |
| astonvilla_seg080 | tiny/full/fp16/box | hold-anchor | 26-50 | 234 | - | 104.75 | 9% | 9% | 12% | - |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | fwd | 1-5 | 123 | 0% | 1.05 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | fwd | 6-12 | 153 | 0% | 1.21 | 87% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | fwd | 13-25 | 263 | 0% | 1.12 | 87% | 98% | 98% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | fwd | 26-50 | 88 | 0% | 1.55 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | bwd | 1-5 | 126 | 0% | 0.98 | 94% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | bwd | 6-12 | 162 | 0% | 1.04 | 91% | 99% | 99% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | bwd | 13-25 | 278 | 0% | 1.19 | 86% | 98% | 98% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | bwd | 26-50 | 131 | 0% | 0.86 | 92% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | hold-anchor | 1-5 | 123 | - | 8.54 | 29% | 55% | 85% | - |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | hold-anchor | 6-12 | 153 | - | 27.17 | 3% | 6% | 27% | - |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | hold-anchor | 13-25 | 263 | - | 55.94 | 1% | 5% | 13% | - |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | hold-anchor | 26-50 | 88 | - | 131.81 | 0% | 0% | 0% | - |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | fwd | 1-5 | 123 | 0% | 1.13 | 94% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | fwd | 6-12 | 153 | 0% | 1.33 | 86% | 98% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | fwd | 13-25 | 263 | 0% | 1.34 | 89% | 98% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | fwd | 26-50 | 88 | 0% | 1.53 | 85% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | bwd | 1-5 | 126 | 0% | 1.16 | 96% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | bwd | 6-12 | 162 | 0% | 1.05 | 93% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | bwd | 13-25 | 278 | 0% | 1.28 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | bwd | 26-50 | 131 | 0% | 0.96 | 95% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | hold-anchor | 1-5 | 123 | - | 8.54 | 29% | 55% | 85% | - |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | hold-anchor | 6-12 | 153 | - | 27.17 | 3% | 6% | 27% | - |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | hold-anchor | 13-25 | 263 | - | 55.94 | 1% | 5% | 13% | - |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | hold-anchor | 26-50 | 88 | - | 131.81 | 0% | 0% | 0% | - |
| bournemouth_1st_seg070 | small/crop512/fp16/box | fwd | 1-5 | 123 | 0% | 1.01 | 92% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | fwd | 6-12 | 153 | 0% | 1.37 | 86% | 98% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | fwd | 13-25 | 263 | 0% | 1.12 | 85% | 97% | 99% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | fwd | 26-50 | 88 | 0% | 1.46 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | bwd | 1-5 | 126 | 0% | 1.02 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | bwd | 6-12 | 162 | 0% | 1.04 | 90% | 96% | 98% | 1% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | bwd | 13-25 | 278 | 0% | 1.38 | 83% | 93% | 94% | 4% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | bwd | 26-50 | 131 | 0% | 0.86 | 93% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | hold-anchor | 1-5 | 123 | - | 8.54 | 29% | 55% | 85% | - |
| bournemouth_1st_seg070 | small/crop512/fp16/box | hold-anchor | 6-12 | 153 | - | 27.17 | 3% | 6% | 27% | - |
| bournemouth_1st_seg070 | small/crop512/fp16/box | hold-anchor | 13-25 | 263 | - | 55.94 | 1% | 5% | 13% | - |
| bournemouth_1st_seg070 | small/crop512/fp16/box | hold-anchor | 26-50 | 88 | - | 131.81 | 0% | 0% | 0% | - |
| bournemouth_1st_seg070 | small/full/fp16/box | fwd | 1-5 | 123 | 0% | 1.14 | 96% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | fwd | 6-12 | 153 | 0% | 1.18 | 94% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | fwd | 13-25 | 263 | 0% | 1.2 | 93% | 98% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | fwd | 26-50 | 88 | 0% | 1.3 | 97% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | bwd | 1-5 | 126 | 0% | 1.14 | 100% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | bwd | 6-12 | 162 | 0% | 1.14 | 95% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | bwd | 13-25 | 278 | 0% | 1.16 | 97% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | bwd | 26-50 | 131 | 0% | 1.01 | 98% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | hold-anchor | 1-5 | 123 | - | 8.54 | 29% | 55% | 85% | - |
| bournemouth_1st_seg070 | small/full/fp16/box | hold-anchor | 6-12 | 153 | - | 27.17 | 3% | 6% | 27% | - |
| bournemouth_1st_seg070 | small/full/fp16/box | hold-anchor | 13-25 | 263 | - | 55.94 | 1% | 5% | 13% | - |
| bournemouth_1st_seg070 | small/full/fp16/box | hold-anchor | 26-50 | 88 | - | 131.81 | 0% | 0% | 0% | - |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | fwd | 1-5 | 123 | 0% | 1.02 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | fwd | 6-12 | 153 | 0% | 1.14 | 87% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | fwd | 13-25 | 263 | 0% | 1.02 | 86% | 97% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | fwd | 26-50 | 88 | 0% | 1.33 | 90% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | bwd | 1-5 | 126 | 0% | 0.97 | 94% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | bwd | 6-12 | 162 | 0% | 1.03 | 92% | 99% | 99% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | bwd | 13-25 | 278 | 0% | 1.23 | 87% | 98% | 99% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | bwd | 26-50 | 131 | 0% | 0.85 | 93% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | hold-anchor | 1-5 | 123 | - | 8.54 | 29% | 55% | 85% | - |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | hold-anchor | 6-12 | 153 | - | 27.17 | 3% | 6% | 27% | - |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | hold-anchor | 13-25 | 263 | - | 55.94 | 1% | 5% | 13% | - |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | hold-anchor | 26-50 | 88 | - | 131.81 | 0% | 0% | 0% | - |
| bournemouth_1st_seg070 | tiny/full/fp16/box | fwd | 1-5 | 123 | 0% | 0.97 | 94% | 98% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | fwd | 6-12 | 153 | 0% | 1.05 | 90% | 97% | 99% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | fwd | 13-25 | 263 | 0% | 0.99 | 92% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | fwd | 26-50 | 88 | 0% | 1.38 | 94% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | bwd | 1-5 | 126 | 0% | 1.01 | 97% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | bwd | 6-12 | 162 | 0% | 0.89 | 94% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | bwd | 13-25 | 278 | 0% | 1.08 | 93% | 97% | 98% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | bwd | 26-50 | 131 | 0% | 0.85 | 98% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | hold-anchor | 1-5 | 123 | - | 8.54 | 29% | 55% | 85% | - |
| bournemouth_1st_seg070 | tiny/full/fp16/box | hold-anchor | 6-12 | 153 | - | 27.17 | 3% | 6% | 27% | - |
| bournemouth_1st_seg070 | tiny/full/fp16/box | hold-anchor | 13-25 | 263 | - | 55.94 | 1% | 5% | 13% | - |
| bournemouth_1st_seg070 | tiny/full/fp16/box | hold-anchor | 26-50 | 88 | - | 131.81 | 0% | 0% | 0% | - |

### Fused vs interpolation by distance to nearest anchor (N=10 / 25 / 50)

| clip | config | N | mode | distance | n held-out | SAM2 absent | median err px | hit@5 | hit@10 | hit@20 | wrong-object (>50px, of present) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| arsenal_example | base_plus/crop512/fp16/box | 10 | interp | 1-5 | 133 | 0% | 4.72 | 52% | 69% | 87% | 3% |
| arsenal_example | base_plus/crop512/fp16/box | 10 | nearest | 1-5 | 133 | 8% | 1.03 | 81% | 82% | 83% | 7% |
| arsenal_example | base_plus/crop512/fp16/box | 25 | interp | 1-5 | 78 | 0% | 4.24 | 54% | 65% | 76% | 1% |
| arsenal_example | base_plus/crop512/fp16/box | 25 | interp | 6-12 | 62 | 0% | 26.51 | 10% | 32% | 45% | 19% |
| arsenal_example | base_plus/crop512/fp16/box | 25 | nearest | 1-5 | 78 | 19% | 1.2 | 73% | 74% | 76% | 3% |
| arsenal_example | base_plus/crop512/fp16/box | 25 | nearest | 6-12 | 62 | 8% | 0.87 | 89% | 89% | 89% | 4% |
| arsenal_example | base_plus/crop512/fp16/box | 50 | interp | 1-5 | 54 | 0% | 4.85 | 52% | 65% | 76% | 4% |
| arsenal_example | base_plus/crop512/fp16/box | 50 | interp | 6-12 | 46 | 0% | 31.74 | 9% | 39% | 39% | 30% |
| arsenal_example | base_plus/crop512/fp16/box | 50 | interp | 13-25 | 43 | 0% | 63.27 | 0% | 14% | 19% | 65% |
| arsenal_example | base_plus/crop512/fp16/box | 50 | nearest | 1-5 | 54 | 22% | 1.14 | 69% | 69% | 70% | 5% |
| arsenal_example | base_plus/crop512/fp16/box | 50 | nearest | 6-12 | 46 | 24% | 0.81 | 76% | 76% | 76% | 0% |
| arsenal_example | base_plus/crop512/fp16/box | 50 | nearest | 13-25 | 43 | 16% | 1.1 | 84% | 84% | 84% | 0% |
| arsenal_example | base_plus/full/fp16/box | 10 | interp | 1-5 | 133 | 0% | 4.72 | 52% | 69% | 87% | 3% |
| arsenal_example | base_plus/full/fp16/box | 10 | nearest | 1-5 | 133 | 4% | 1.75 | 76% | 80% | 81% | 9% |
| arsenal_example | base_plus/full/fp16/box | 25 | interp | 1-5 | 78 | 0% | 4.24 | 54% | 65% | 76% | 1% |
| arsenal_example | base_plus/full/fp16/box | 25 | interp | 6-12 | 62 | 0% | 26.51 | 10% | 32% | 45% | 19% |
| arsenal_example | base_plus/full/fp16/box | 25 | nearest | 1-5 | 78 | 6% | 2.03 | 69% | 73% | 77% | 12% |
| arsenal_example | base_plus/full/fp16/box | 25 | nearest | 6-12 | 62 | 10% | 1.38 | 77% | 79% | 84% | 7% |
| arsenal_example | base_plus/full/fp16/box | 50 | interp | 1-5 | 54 | 0% | 4.85 | 52% | 65% | 76% | 4% |
| arsenal_example | base_plus/full/fp16/box | 50 | interp | 6-12 | 46 | 0% | 31.74 | 9% | 39% | 39% | 30% |
| arsenal_example | base_plus/full/fp16/box | 50 | interp | 13-25 | 43 | 0% | 63.27 | 0% | 14% | 19% | 65% |
| arsenal_example | base_plus/full/fp16/box | 50 | nearest | 1-5 | 54 | 9% | 2.03 | 67% | 70% | 70% | 14% |
| arsenal_example | base_plus/full/fp16/box | 50 | nearest | 6-12 | 46 | 13% | 1.29 | 67% | 70% | 70% | 20% |
| arsenal_example | base_plus/full/fp16/box | 50 | nearest | 13-25 | 43 | 0% | 3.41 | 51% | 56% | 56% | 44% |
| arsenal_example | small/crop512/fp16/box | 10 | interp | 1-5 | 133 | 0% | 4.72 | 52% | 69% | 87% | 3% |
| arsenal_example | small/crop512/fp16/box | 10 | nearest | 1-5 | 133 | 8% | 1.15 | 81% | 82% | 83% | 7% |
| arsenal_example | small/crop512/fp16/box | 25 | interp | 1-5 | 78 | 0% | 4.24 | 54% | 65% | 76% | 1% |
| arsenal_example | small/crop512/fp16/box | 25 | interp | 6-12 | 62 | 0% | 26.51 | 10% | 32% | 45% | 19% |
| arsenal_example | small/crop512/fp16/box | 25 | nearest | 1-5 | 78 | 15% | 1.24 | 76% | 76% | 77% | 6% |
| arsenal_example | small/crop512/fp16/box | 25 | nearest | 6-12 | 62 | 6% | 1.02 | 92% | 94% | 94% | 0% |
| arsenal_example | small/crop512/fp16/box | 50 | interp | 1-5 | 54 | 0% | 4.85 | 52% | 65% | 76% | 4% |
| arsenal_example | small/crop512/fp16/box | 50 | interp | 6-12 | 46 | 0% | 31.74 | 9% | 39% | 39% | 30% |
| arsenal_example | small/crop512/fp16/box | 50 | interp | 13-25 | 43 | 0% | 63.27 | 0% | 14% | 19% | 65% |
| arsenal_example | small/crop512/fp16/box | 50 | nearest | 1-5 | 54 | 20% | 1.21 | 69% | 69% | 70% | 7% |
| arsenal_example | small/crop512/fp16/box | 50 | nearest | 6-12 | 46 | 17% | 0.88 | 80% | 83% | 83% | 0% |
| arsenal_example | small/crop512/fp16/box | 50 | nearest | 13-25 | 43 | 16% | 1.02 | 84% | 84% | 84% | 0% |
| arsenal_example | small/full/fp16/box | 10 | interp | 1-5 | 133 | 0% | 4.72 | 52% | 69% | 87% | 3% |
| arsenal_example | small/full/fp16/box | 10 | nearest | 1-5 | 133 | 7% | 1.91 | 74% | 75% | 79% | 11% |
| arsenal_example | small/full/fp16/box | 25 | interp | 1-5 | 78 | 0% | 4.24 | 54% | 65% | 76% | 1% |
| arsenal_example | small/full/fp16/box | 25 | interp | 6-12 | 62 | 0% | 26.51 | 10% | 32% | 45% | 19% |
| arsenal_example | small/full/fp16/box | 25 | nearest | 1-5 | 78 | 8% | 2.51 | 67% | 72% | 76% | 15% |
| arsenal_example | small/full/fp16/box | 25 | nearest | 6-12 | 62 | 11% | 1.65 | 65% | 84% | 84% | 5% |
| arsenal_example | small/full/fp16/box | 50 | interp | 1-5 | 54 | 0% | 4.85 | 52% | 65% | 76% | 4% |
| arsenal_example | small/full/fp16/box | 50 | interp | 6-12 | 46 | 0% | 31.74 | 9% | 39% | 39% | 30% |
| arsenal_example | small/full/fp16/box | 50 | interp | 13-25 | 43 | 0% | 63.27 | 0% | 14% | 19% | 65% |
| arsenal_example | small/full/fp16/box | 50 | nearest | 1-5 | 54 | 9% | 2.54 | 67% | 69% | 70% | 18% |
| arsenal_example | small/full/fp16/box | 50 | nearest | 6-12 | 46 | 15% | 1.97 | 54% | 70% | 70% | 18% |
| arsenal_example | small/full/fp16/box | 50 | nearest | 13-25 | 43 | 28% | 2.14 | 47% | 56% | 56% | 23% |
| arsenal_example | tiny/crop512/fp16/box | 10 | interp | 1-5 | 133 | 0% | 4.72 | 52% | 69% | 87% | 3% |
| arsenal_example | tiny/crop512/fp16/box | 10 | nearest | 1-5 | 133 | 6% | 1.12 | 82% | 82% | 83% | 9% |
| arsenal_example | tiny/crop512/fp16/box | 25 | interp | 1-5 | 78 | 0% | 4.24 | 54% | 65% | 76% | 1% |
| arsenal_example | tiny/crop512/fp16/box | 25 | interp | 6-12 | 62 | 0% | 26.51 | 10% | 32% | 45% | 19% |
| arsenal_example | tiny/crop512/fp16/box | 25 | nearest | 1-5 | 78 | 12% | 1.14 | 74% | 74% | 76% | 10% |
| arsenal_example | tiny/crop512/fp16/box | 25 | nearest | 6-12 | 62 | 15% | 0.98 | 82% | 84% | 84% | 2% |
| arsenal_example | tiny/crop512/fp16/box | 50 | interp | 1-5 | 54 | 0% | 4.85 | 52% | 65% | 76% | 4% |
| arsenal_example | tiny/crop512/fp16/box | 50 | interp | 6-12 | 46 | 0% | 31.74 | 9% | 39% | 39% | 30% |
| arsenal_example | tiny/crop512/fp16/box | 50 | interp | 13-25 | 43 | 0% | 63.27 | 0% | 14% | 19% | 65% |
| arsenal_example | tiny/crop512/fp16/box | 50 | nearest | 1-5 | 54 | 17% | 1.14 | 69% | 69% | 70% | 11% |
| arsenal_example | tiny/crop512/fp16/box | 50 | nearest | 6-12 | 46 | 20% | 0.95 | 67% | 70% | 70% | 14% |
| arsenal_example | tiny/crop512/fp16/box | 50 | nearest | 13-25 | 43 | 30% | 1.39 | 56% | 56% | 56% | 20% |
| arsenal_example | tiny/full/fp16/box | 10 | interp | 1-5 | 133 | 0% | 4.72 | 52% | 69% | 87% | 3% |
| arsenal_example | tiny/full/fp16/box | 10 | nearest | 1-5 | 133 | 5% | 1.99 | 71% | 75% | 80% | 12% |
| arsenal_example | tiny/full/fp16/box | 25 | interp | 1-5 | 78 | 0% | 4.24 | 54% | 65% | 76% | 1% |
| arsenal_example | tiny/full/fp16/box | 25 | interp | 6-12 | 62 | 0% | 26.51 | 10% | 32% | 45% | 19% |
| arsenal_example | tiny/full/fp16/box | 25 | nearest | 1-5 | 78 | 6% | 3.1 | 64% | 72% | 77% | 15% |
| arsenal_example | tiny/full/fp16/box | 25 | nearest | 6-12 | 62 | 8% | 1.96 | 63% | 81% | 84% | 9% |
| arsenal_example | tiny/full/fp16/box | 50 | interp | 1-5 | 54 | 0% | 4.85 | 52% | 65% | 76% | 4% |
| arsenal_example | tiny/full/fp16/box | 50 | interp | 6-12 | 46 | 0% | 31.74 | 9% | 39% | 39% | 30% |
| arsenal_example | tiny/full/fp16/box | 50 | interp | 13-25 | 43 | 0% | 63.27 | 0% | 14% | 19% | 65% |
| arsenal_example | tiny/full/fp16/box | 50 | nearest | 1-5 | 54 | 9% | 3.58 | 63% | 69% | 70% | 18% |
| arsenal_example | tiny/full/fp16/box | 50 | nearest | 6-12 | 46 | 11% | 4.15 | 50% | 65% | 70% | 22% |
| arsenal_example | tiny/full/fp16/box | 50 | nearest | 13-25 | 43 | 0% | 6.36 | 47% | 56% | 56% | 44% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 10 | interp | 1-5 | 505 | 0% | 2.01 | 74% | 89% | 96% | 1% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 10 | nearest | 1-5 | 505 | 1% | 1.06 | 93% | 97% | 97% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 25 | interp | 1-5 | 239 | 0% | 5.26 | 49% | 66% | 80% | 4% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 25 | interp | 6-12 | 300 | 0% | 8.56 | 37% | 54% | 67% | 13% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 25 | nearest | 1-5 | 239 | 0% | 1.19 | 91% | 95% | 97% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 25 | nearest | 6-12 | 300 | 2% | 1.09 | 84% | 89% | 93% | 2% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | interp | 1-5 | 129 | 0% | 8.24 | 38% | 57% | 74% | 6% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | interp | 6-12 | 154 | 0% | 18.33 | 30% | 37% | 55% | 28% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | interp | 13-25 | 267 | 0% | 22.93 | 29% | 37% | 44% | 41% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | nearest | 1-5 | 129 | 1% | 1.16 | 91% | 95% | 95% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | nearest | 6-12 | 154 | 1% | 1.06 | 85% | 92% | 95% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | nearest | 13-25 | 267 | 1% | 1.18 | 85% | 91% | 94% | 5% |
| astonvilla_seg080 | base_plus/full/fp16/box | 10 | interp | 1-5 | 505 | 0% | 2.01 | 74% | 89% | 96% | 1% |
| astonvilla_seg080 | base_plus/full/fp16/box | 10 | nearest | 1-5 | 505 | 0% | 1.12 | 92% | 97% | 98% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 25 | interp | 1-5 | 239 | 0% | 5.26 | 49% | 66% | 80% | 4% |
| astonvilla_seg080 | base_plus/full/fp16/box | 25 | interp | 6-12 | 300 | 0% | 8.56 | 37% | 54% | 67% | 13% |
| astonvilla_seg080 | base_plus/full/fp16/box | 25 | nearest | 1-5 | 239 | 0% | 1.3 | 89% | 95% | 97% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 25 | nearest | 6-12 | 300 | 0% | 1.1 | 89% | 93% | 97% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | interp | 1-5 | 129 | 0% | 8.24 | 38% | 57% | 74% | 6% |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | interp | 6-12 | 154 | 0% | 18.33 | 30% | 37% | 55% | 28% |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | interp | 13-25 | 267 | 0% | 22.93 | 29% | 37% | 44% | 41% |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | nearest | 1-5 | 129 | 0% | 1.25 | 85% | 91% | 94% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | nearest | 6-12 | 154 | 1% | 1.07 | 87% | 91% | 94% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | nearest | 13-25 | 267 | 1% | 1.27 | 84% | 89% | 94% | 2% |
| astonvilla_seg080 | small/crop512/fp16/box | 10 | interp | 1-5 | 505 | 0% | 2.01 | 74% | 89% | 96% | 1% |
| astonvilla_seg080 | small/crop512/fp16/box | 10 | nearest | 1-5 | 505 | 1% | 1.34 | 93% | 97% | 97% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 25 | interp | 1-5 | 239 | 0% | 5.26 | 49% | 66% | 80% | 4% |
| astonvilla_seg080 | small/crop512/fp16/box | 25 | interp | 6-12 | 300 | 0% | 8.56 | 37% | 54% | 67% | 13% |
| astonvilla_seg080 | small/crop512/fp16/box | 25 | nearest | 1-5 | 239 | 0% | 1.5 | 93% | 95% | 97% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 25 | nearest | 6-12 | 300 | 1% | 1.34 | 89% | 94% | 97% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | interp | 1-5 | 129 | 0% | 8.24 | 38% | 57% | 74% | 6% |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | interp | 6-12 | 154 | 0% | 18.33 | 30% | 37% | 55% | 28% |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | interp | 13-25 | 267 | 0% | 22.93 | 29% | 37% | 44% | 41% |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | nearest | 1-5 | 129 | 0% | 1.47 | 92% | 95% | 95% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | nearest | 6-12 | 154 | 1% | 1.31 | 86% | 92% | 95% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | nearest | 13-25 | 267 | 1% | 1.52 | 83% | 90% | 95% | 5% |
| astonvilla_seg080 | small/full/fp16/box | 10 | interp | 1-5 | 505 | 0% | 2.01 | 74% | 89% | 96% | 1% |
| astonvilla_seg080 | small/full/fp16/box | 10 | nearest | 1-5 | 505 | 0% | 1.46 | 92% | 97% | 98% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 25 | interp | 1-5 | 239 | 0% | 5.26 | 49% | 66% | 80% | 4% |
| astonvilla_seg080 | small/full/fp16/box | 25 | interp | 6-12 | 300 | 0% | 8.56 | 37% | 54% | 67% | 13% |
| astonvilla_seg080 | small/full/fp16/box | 25 | nearest | 1-5 | 239 | 0% | 1.59 | 89% | 95% | 96% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 25 | nearest | 6-12 | 300 | 1% | 1.45 | 87% | 93% | 96% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 50 | interp | 1-5 | 129 | 0% | 8.24 | 38% | 57% | 74% | 6% |
| astonvilla_seg080 | small/full/fp16/box | 50 | interp | 6-12 | 154 | 0% | 18.33 | 30% | 37% | 55% | 28% |
| astonvilla_seg080 | small/full/fp16/box | 50 | interp | 13-25 | 267 | 0% | 22.93 | 29% | 37% | 44% | 41% |
| astonvilla_seg080 | small/full/fp16/box | 50 | nearest | 1-5 | 129 | 0% | 1.59 | 86% | 92% | 93% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 50 | nearest | 6-12 | 154 | 1% | 1.42 | 86% | 93% | 95% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 50 | nearest | 13-25 | 267 | 1% | 1.6 | 85% | 90% | 94% | 5% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 10 | interp | 1-5 | 505 | 0% | 2.01 | 74% | 89% | 96% | 1% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 10 | nearest | 1-5 | 505 | 1% | 1.21 | 93% | 97% | 97% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 25 | interp | 1-5 | 239 | 0% | 5.26 | 49% | 66% | 80% | 4% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 25 | interp | 6-12 | 300 | 0% | 8.56 | 37% | 54% | 67% | 13% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 25 | nearest | 1-5 | 239 | 0% | 1.32 | 92% | 96% | 97% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 25 | nearest | 6-12 | 300 | 1% | 1.19 | 88% | 92% | 95% | 1% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | interp | 1-5 | 129 | 0% | 8.24 | 38% | 57% | 74% | 6% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | interp | 6-12 | 154 | 0% | 18.33 | 30% | 37% | 55% | 28% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | interp | 13-25 | 267 | 0% | 22.93 | 29% | 37% | 44% | 41% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | nearest | 1-5 | 129 | 0% | 1.27 | 91% | 95% | 95% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | nearest | 6-12 | 154 | 1% | 1.15 | 86% | 93% | 95% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | nearest | 13-25 | 267 | 0% | 1.35 | 82% | 91% | 95% | 5% |
| astonvilla_seg080 | tiny/full/fp16/box | 10 | interp | 1-5 | 505 | 0% | 2.01 | 74% | 89% | 96% | 1% |
| astonvilla_seg080 | tiny/full/fp16/box | 10 | nearest | 1-5 | 505 | 0% | 1.29 | 93% | 97% | 98% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 25 | interp | 1-5 | 239 | 0% | 5.26 | 49% | 66% | 80% | 4% |
| astonvilla_seg080 | tiny/full/fp16/box | 25 | interp | 6-12 | 300 | 0% | 8.56 | 37% | 54% | 67% | 13% |
| astonvilla_seg080 | tiny/full/fp16/box | 25 | nearest | 1-5 | 239 | 0% | 1.34 | 87% | 95% | 96% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 25 | nearest | 6-12 | 300 | 0% | 1.3 | 91% | 96% | 99% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | interp | 1-5 | 129 | 0% | 8.24 | 38% | 57% | 74% | 6% |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | interp | 6-12 | 154 | 0% | 18.33 | 30% | 37% | 55% | 28% |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | interp | 13-25 | 267 | 0% | 22.93 | 29% | 37% | 44% | 41% |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | nearest | 1-5 | 129 | 0% | 1.24 | 84% | 91% | 93% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | nearest | 6-12 | 154 | 0% | 1.28 | 91% | 95% | 99% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | nearest | 13-25 | 267 | 0% | 1.45 | 88% | 93% | 98% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 10 | interp | 1-5 | 195 | 0% | 1.68 | 91% | 95% | 97% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 10 | nearest | 1-5 | 195 | 0% | 1.05 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 25 | interp | 1-5 | 95 | 0% | 3.24 | 67% | 89% | 98% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 25 | interp | 6-12 | 113 | 0% | 10.47 | 20% | 48% | 85% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 25 | nearest | 1-5 | 95 | 0% | 0.89 | 95% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 25 | nearest | 6-12 | 113 | 0% | 1.36 | 86% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 50 | interp | 1-5 | 55 | 0% | 6.61 | 42% | 65% | 87% | 2% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 50 | interp | 6-12 | 57 | 0% | 27.51 | 2% | 4% | 26% | 25% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 50 | interp | 13-25 | 100 | 0% | 40.83 | 0% | 0% | 5% | 25% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 50 | nearest | 1-5 | 55 | 0% | 0.85 | 93% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 50 | nearest | 6-12 | 57 | 0% | 1.17 | 86% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 50 | nearest | 13-25 | 100 | 0% | 1.28 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 10 | interp | 1-5 | 195 | 0% | 1.68 | 91% | 95% | 97% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 10 | nearest | 1-5 | 195 | 0% | 1.21 | 94% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 25 | interp | 1-5 | 95 | 0% | 3.24 | 67% | 89% | 98% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 25 | interp | 6-12 | 113 | 0% | 10.47 | 20% | 48% | 85% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 25 | nearest | 1-5 | 95 | 0% | 0.97 | 99% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 25 | nearest | 6-12 | 113 | 0% | 1.37 | 89% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 50 | interp | 1-5 | 55 | 0% | 6.61 | 42% | 65% | 87% | 2% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 50 | interp | 6-12 | 57 | 0% | 27.51 | 2% | 4% | 26% | 25% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 50 | interp | 13-25 | 100 | 0% | 40.83 | 0% | 0% | 5% | 25% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 50 | nearest | 1-5 | 55 | 0% | 0.97 | 100% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 50 | nearest | 6-12 | 57 | 0% | 1.37 | 93% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 50 | nearest | 13-25 | 100 | 0% | 1.41 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 10 | interp | 1-5 | 195 | 0% | 1.68 | 91% | 95% | 97% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 10 | nearest | 1-5 | 195 | 0% | 1.19 | 90% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 25 | interp | 1-5 | 95 | 0% | 3.24 | 67% | 89% | 98% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 25 | interp | 6-12 | 113 | 0% | 10.47 | 20% | 48% | 85% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 25 | nearest | 1-5 | 95 | 0% | 0.87 | 95% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 25 | nearest | 6-12 | 113 | 0% | 1.37 | 88% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 50 | interp | 1-5 | 55 | 0% | 6.61 | 42% | 65% | 87% | 2% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 50 | interp | 6-12 | 57 | 0% | 27.51 | 2% | 4% | 26% | 25% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 50 | interp | 13-25 | 100 | 0% | 40.83 | 0% | 0% | 5% | 25% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 50 | nearest | 1-5 | 55 | 0% | 0.81 | 93% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 50 | nearest | 6-12 | 57 | 0% | 1.04 | 89% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 50 | nearest | 13-25 | 100 | 0% | 1.37 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 10 | interp | 1-5 | 195 | 0% | 1.68 | 91% | 95% | 97% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 10 | nearest | 1-5 | 195 | 0% | 1.15 | 98% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 25 | interp | 1-5 | 95 | 0% | 3.24 | 67% | 89% | 98% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 25 | interp | 6-12 | 113 | 0% | 10.47 | 20% | 48% | 85% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 25 | nearest | 1-5 | 95 | 0% | 1.03 | 98% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 25 | nearest | 6-12 | 113 | 0% | 1.28 | 95% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 50 | interp | 1-5 | 55 | 0% | 6.61 | 42% | 65% | 87% | 2% |
| bournemouth_1st_seg070 | small/full/fp16/box | 50 | interp | 6-12 | 57 | 0% | 27.51 | 2% | 4% | 26% | 25% |
| bournemouth_1st_seg070 | small/full/fp16/box | 50 | interp | 13-25 | 100 | 0% | 40.83 | 0% | 0% | 5% | 25% |
| bournemouth_1st_seg070 | small/full/fp16/box | 50 | nearest | 1-5 | 55 | 0% | 0.93 | 100% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 50 | nearest | 6-12 | 57 | 0% | 1.2 | 95% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 50 | nearest | 13-25 | 100 | 0% | 1.24 | 96% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 10 | interp | 1-5 | 195 | 0% | 1.68 | 91% | 95% | 97% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 10 | nearest | 1-5 | 195 | 0% | 1.03 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 25 | interp | 1-5 | 95 | 0% | 3.24 | 67% | 89% | 98% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 25 | interp | 6-12 | 113 | 0% | 10.47 | 20% | 48% | 85% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 25 | nearest | 1-5 | 95 | 0% | 0.82 | 95% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 25 | nearest | 6-12 | 113 | 0% | 1.22 | 88% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 50 | interp | 1-5 | 55 | 0% | 6.61 | 42% | 65% | 87% | 2% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 50 | interp | 6-12 | 57 | 0% | 27.51 | 2% | 4% | 26% | 25% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 50 | interp | 13-25 | 100 | 0% | 40.83 | 0% | 0% | 5% | 25% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 50 | nearest | 1-5 | 55 | 0% | 0.74 | 93% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 50 | nearest | 6-12 | 57 | 0% | 1.03 | 89% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 50 | nearest | 13-25 | 100 | 0% | 1.28 | 91% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 10 | interp | 1-5 | 195 | 0% | 1.68 | 91% | 95% | 97% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 10 | nearest | 1-5 | 195 | 0% | 1.03 | 95% | 99% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 25 | interp | 1-5 | 95 | 0% | 3.24 | 67% | 89% | 98% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 25 | interp | 6-12 | 113 | 0% | 10.47 | 20% | 48% | 85% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 25 | nearest | 1-5 | 95 | 0% | 0.73 | 95% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 25 | nearest | 6-12 | 113 | 0% | 1.07 | 95% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 50 | interp | 1-5 | 55 | 0% | 6.61 | 42% | 65% | 87% | 2% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 50 | interp | 6-12 | 57 | 0% | 27.51 | 2% | 4% | 26% | 25% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 50 | interp | 13-25 | 100 | 0% | 40.83 | 0% | 0% | 5% | 25% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 50 | nearest | 1-5 | 55 | 0% | 0.72 | 100% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 50 | nearest | 6-12 | 57 | 0% | 1.02 | 96% | 100% | 100% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 50 | nearest | 13-25 | 100 | 0% | 1.2 | 96% | 100% | 100% | 0% |

### no_ball frames (false positives) and not_broadcast frames

| clip | config | no_ball frames | SAM2 predicts a ball (FP) | not_broadcast frames | SAM2 predicts a ball |
|---|---|---|---|---|---|
| arsenal_example | base_plus/crop512/fp16/box | 0 | 0 | 0 | 0 |
| arsenal_example | base_plus/full/fp16/box | 0 | 0 | 0 | 0 |
| arsenal_example | small/crop512/fp16/box | 0 | 0 | 0 | 0 |
| arsenal_example | small/full/fp16/box | 0 | 0 | 0 | 0 |
| arsenal_example | tiny/crop512/fp16/box | 0 | 0 | 0 | 0 |
| arsenal_example | tiny/full/fp16/box | 0 | 0 | 0 | 0 |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 5 | 2 | 0 | 0 |
| astonvilla_seg080 | base_plus/full/fp16/box | 5 | 3 | 0 | 0 |
| astonvilla_seg080 | small/crop512/fp16/box | 5 | 2 | 0 | 0 |
| astonvilla_seg080 | small/full/fp16/box | 5 | 4 | 0 | 0 |
| astonvilla_seg080 | tiny/crop512/fp16/box | 5 | 2 | 0 | 0 |
| astonvilla_seg080 | tiny/full/fp16/box | 5 | 3 | 0 | 0 |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 0 | 0 | 4 | 0 |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 0 | 0 | 4 | 4 |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 0 | 0 | 4 | 0 |
| bournemouth_1st_seg070 | small/full/fp16/box | 0 | 0 | 4 | 4 |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 0 | 0 | 4 | 0 |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 0 | 0 | 4 | 4 |

### Failure modes of the fused-nearest prediction (all held-out frames)

| clip | config | N | ball speed | n | hit<=10 | near 10-20 | wrong, on player/ref box | wrong, elsewhere | absent |
|---|---|---|---|---|---|---|---|---|---|
| arsenal_example | base_plus/crop512/fp16/box | 10 | slow(<8px/f) | 92 | 98% | 0% | 0% | 0% | 2% |
| arsenal_example | base_plus/crop512/fp16/box | 10 | mid(8-20) | 31 | 61% | 3% | 0% | 19% | 16% |
| arsenal_example | base_plus/crop512/fp16/box | 10 | fast(>20) | 10 | 0% | 0% | 0% | 60% | 40% |
| arsenal_example | base_plus/crop512/fp16/box | 10 | ALL | 133 | 82% | 1% | 0% | 9% | 8% |
| arsenal_example | base_plus/crop512/fp16/box | 25 | slow(<8px/f) | 98 | 96% | 0% | 0% | 0% | 4% |
| arsenal_example | base_plus/crop512/fp16/box | 25 | mid(8-20) | 32 | 59% | 3% | 0% | 9% | 28% |
| arsenal_example | base_plus/crop512/fp16/box | 25 | fast(>20) | 10 | 0% | 0% | 0% | 30% | 70% |
| arsenal_example | base_plus/crop512/fp16/box | 25 | ALL | 140 | 81% | 1% | 0% | 4% | 14% |
| arsenal_example | base_plus/crop512/fp16/box | 50 | slow(<8px/f) | 100 | 91% | 0% | 0% | 0% | 9% |
| arsenal_example | base_plus/crop512/fp16/box | 50 | mid(8-20) | 33 | 52% | 3% | 0% | 9% | 36% |
| arsenal_example | base_plus/crop512/fp16/box | 50 | fast(>20) | 10 | 0% | 0% | 0% | 10% | 90% |
| arsenal_example | base_plus/crop512/fp16/box | 50 | ALL | 143 | 76% | 1% | 0% | 3% | 21% |
| arsenal_example | base_plus/full/fp16/box | 10 | slow(<8px/f) | 92 | 97% | 0% | 1% | 0% | 2% |
| arsenal_example | base_plus/full/fp16/box | 10 | mid(8-20) | 31 | 55% | 3% | 3% | 32% | 6% |
| arsenal_example | base_plus/full/fp16/box | 10 | fast(>20) | 10 | 0% | 10% | 0% | 80% | 10% |
| arsenal_example | base_plus/full/fp16/box | 10 | ALL | 133 | 80% | 2% | 2% | 14% | 4% |
| arsenal_example | base_plus/full/fp16/box | 25 | slow(<8px/f) | 98 | 89% | 4% | 0% | 1% | 6% |
| arsenal_example | base_plus/full/fp16/box | 25 | mid(8-20) | 32 | 59% | 6% | 3% | 19% | 12% |
| arsenal_example | base_plus/full/fp16/box | 25 | fast(>20) | 10 | 0% | 0% | 20% | 70% | 10% |
| arsenal_example | base_plus/full/fp16/box | 25 | ALL | 140 | 76% | 4% | 2% | 10% | 8% |
| arsenal_example | base_plus/full/fp16/box | 50 | slow(<8px/f) | 100 | 78% | 0% | 0% | 16% | 6% |
| arsenal_example | base_plus/full/fp16/box | 50 | mid(8-20) | 33 | 48% | 0% | 0% | 39% | 12% |
| arsenal_example | base_plus/full/fp16/box | 50 | fast(>20) | 10 | 0% | 0% | 0% | 90% | 10% |
| arsenal_example | base_plus/full/fp16/box | 50 | ALL | 143 | 66% | 0% | 0% | 27% | 8% |
| arsenal_example | small/crop512/fp16/box | 10 | slow(<8px/f) | 92 | 98% | 0% | 0% | 0% | 2% |
| arsenal_example | small/crop512/fp16/box | 10 | mid(8-20) | 31 | 61% | 3% | 0% | 23% | 13% |
| arsenal_example | small/crop512/fp16/box | 10 | fast(>20) | 10 | 0% | 0% | 0% | 60% | 40% |
| arsenal_example | small/crop512/fp16/box | 10 | ALL | 133 | 82% | 1% | 0% | 10% | 8% |
| arsenal_example | small/crop512/fp16/box | 25 | slow(<8px/f) | 98 | 98% | 0% | 0% | 0% | 2% |
| arsenal_example | small/crop512/fp16/box | 25 | mid(8-20) | 32 | 66% | 3% | 0% | 12% | 19% |
| arsenal_example | small/crop512/fp16/box | 25 | fast(>20) | 10 | 0% | 0% | 0% | 20% | 80% |
| arsenal_example | small/crop512/fp16/box | 25 | ALL | 140 | 84% | 1% | 0% | 4% | 11% |
| arsenal_example | small/crop512/fp16/box | 50 | slow(<8px/f) | 100 | 93% | 0% | 0% | 0% | 7% |
| arsenal_example | small/crop512/fp16/box | 50 | mid(8-20) | 33 | 55% | 3% | 0% | 12% | 30% |
| arsenal_example | small/crop512/fp16/box | 50 | fast(>20) | 10 | 0% | 0% | 0% | 10% | 90% |
| arsenal_example | small/crop512/fp16/box | 50 | ALL | 143 | 78% | 1% | 0% | 3% | 18% |
| arsenal_example | small/full/fp16/box | 10 | slow(<8px/f) | 92 | 92% | 4% | 0% | 0% | 3% |
| arsenal_example | small/full/fp16/box | 10 | mid(8-20) | 31 | 48% | 3% | 0% | 32% | 16% |
| arsenal_example | small/full/fp16/box | 10 | fast(>20) | 10 | 0% | 0% | 0% | 90% | 10% |
| arsenal_example | small/full/fp16/box | 10 | ALL | 133 | 75% | 4% | 0% | 14% | 7% |
| arsenal_example | small/full/fp16/box | 25 | slow(<8px/f) | 98 | 91% | 1% | 0% | 0% | 8% |
| arsenal_example | small/full/fp16/box | 25 | mid(8-20) | 32 | 59% | 6% | 0% | 22% | 12% |
| arsenal_example | small/full/fp16/box | 25 | fast(>20) | 10 | 0% | 0% | 20% | 70% | 10% |
| arsenal_example | small/full/fp16/box | 25 | ALL | 140 | 77% | 2% | 1% | 10% | 9% |
| arsenal_example | small/full/fp16/box | 50 | slow(<8px/f) | 100 | 78% | 0% | 0% | 5% | 17% |
| arsenal_example | small/full/fp16/box | 50 | mid(8-20) | 33 | 45% | 3% | 0% | 33% | 18% |
| arsenal_example | small/full/fp16/box | 50 | fast(>20) | 10 | 0% | 0% | 0% | 90% | 10% |
| arsenal_example | small/full/fp16/box | 50 | ALL | 143 | 65% | 1% | 0% | 17% | 17% |
| arsenal_example | tiny/crop512/fp16/box | 10 | slow(<8px/f) | 92 | 98% | 0% | 0% | 0% | 2% |
| arsenal_example | tiny/crop512/fp16/box | 10 | mid(8-20) | 31 | 61% | 3% | 0% | 23% | 13% |
| arsenal_example | tiny/crop512/fp16/box | 10 | fast(>20) | 10 | 0% | 0% | 0% | 80% | 20% |
| arsenal_example | tiny/crop512/fp16/box | 10 | ALL | 133 | 82% | 1% | 0% | 11% | 6% |
| arsenal_example | tiny/crop512/fp16/box | 25 | slow(<8px/f) | 98 | 93% | 0% | 0% | 0% | 7% |
| arsenal_example | tiny/crop512/fp16/box | 25 | mid(8-20) | 32 | 59% | 3% | 3% | 16% | 19% |
| arsenal_example | tiny/crop512/fp16/box | 25 | fast(>20) | 10 | 0% | 0% | 10% | 40% | 50% |
| arsenal_example | tiny/crop512/fp16/box | 25 | ALL | 140 | 79% | 1% | 1% | 6% | 13% |
| arsenal_example | tiny/crop512/fp16/box | 50 | slow(<8px/f) | 100 | 78% | 0% | 0% | 4% | 18% |
| arsenal_example | tiny/crop512/fp16/box | 50 | mid(8-20) | 33 | 45% | 3% | 0% | 27% | 24% |
| arsenal_example | tiny/crop512/fp16/box | 50 | fast(>20) | 10 | 0% | 0% | 0% | 50% | 50% |
| arsenal_example | tiny/crop512/fp16/box | 50 | ALL | 143 | 65% | 1% | 0% | 13% | 22% |
| arsenal_example | tiny/full/fp16/box | 10 | slow(<8px/f) | 92 | 92% | 4% | 1% | 0% | 2% |
| arsenal_example | tiny/full/fp16/box | 10 | mid(8-20) | 31 | 48% | 3% | 6% | 32% | 10% |
| arsenal_example | tiny/full/fp16/box | 10 | fast(>20) | 10 | 0% | 10% | 0% | 80% | 10% |
| arsenal_example | tiny/full/fp16/box | 10 | ALL | 133 | 75% | 5% | 2% | 14% | 5% |
| arsenal_example | tiny/full/fp16/box | 25 | slow(<8px/f) | 98 | 90% | 3% | 2% | 0% | 5% |
| arsenal_example | tiny/full/fp16/box | 25 | mid(8-20) | 32 | 56% | 9% | 3% | 19% | 12% |
| arsenal_example | tiny/full/fp16/box | 25 | fast(>20) | 10 | 0% | 0% | 20% | 70% | 10% |
| arsenal_example | tiny/full/fp16/box | 25 | ALL | 140 | 76% | 4% | 4% | 9% | 7% |
| arsenal_example | tiny/full/fp16/box | 50 | slow(<8px/f) | 100 | 76% | 2% | 12% | 5% | 5% |
| arsenal_example | tiny/full/fp16/box | 50 | mid(8-20) | 33 | 45% | 3% | 6% | 33% | 12% |
| arsenal_example | tiny/full/fp16/box | 50 | fast(>20) | 10 | 0% | 0% | 0% | 90% | 10% |
| arsenal_example | tiny/full/fp16/box | 50 | ALL | 143 | 64% | 2% | 10% | 17% | 7% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 10 | slow(<8px/f) | 387 | 97% | 1% | 0% | 2% | 1% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 10 | mid(8-20) | 103 | 96% | 0% | 1% | 0% | 3% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 10 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 10 | ALL | 505 | 97% | 0% | 0% | 1% | 1% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 25 | slow(<8px/f) | 413 | 94% | 3% | 0% | 1% | 1% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 25 | mid(8-20) | 111 | 85% | 1% | 13% | 0% | 2% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 25 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 25 | ALL | 539 | 92% | 3% | 3% | 1% | 1% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | slow(<8px/f) | 421 | 94% | 4% | 0% | 1% | 1% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | mid(8-20) | 114 | 83% | 0% | 15% | 0% | 2% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 50 | ALL | 550 | 92% | 3% | 3% | 1% | 1% |
| astonvilla_seg080 | base_plus/full/fp16/box | 10 | slow(<8px/f) | 387 | 98% | 1% | 0% | 2% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 10 | mid(8-20) | 103 | 93% | 5% | 2% | 0% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 10 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 10 | ALL | 505 | 97% | 1% | 1% | 1% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 25 | slow(<8px/f) | 413 | 95% | 4% | 0% | 1% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 25 | mid(8-20) | 111 | 89% | 3% | 8% | 0% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 25 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 25 | ALL | 539 | 94% | 3% | 2% | 1% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | slow(<8px/f) | 421 | 93% | 5% | 0% | 1% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | mid(8-20) | 114 | 78% | 3% | 18% | 0% | 2% |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | base_plus/full/fp16/box | 50 | ALL | 550 | 90% | 4% | 4% | 1% | 1% |
| astonvilla_seg080 | small/crop512/fp16/box | 10 | slow(<8px/f) | 387 | 97% | 1% | 0% | 2% | 1% |
| astonvilla_seg080 | small/crop512/fp16/box | 10 | mid(8-20) | 103 | 96% | 0% | 1% | 0% | 3% |
| astonvilla_seg080 | small/crop512/fp16/box | 10 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 10 | ALL | 505 | 97% | 0% | 0% | 1% | 1% |
| astonvilla_seg080 | small/crop512/fp16/box | 25 | slow(<8px/f) | 413 | 95% | 3% | 0% | 1% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 25 | mid(8-20) | 111 | 93% | 1% | 5% | 0% | 1% |
| astonvilla_seg080 | small/crop512/fp16/box | 25 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 25 | ALL | 539 | 94% | 3% | 1% | 1% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | slow(<8px/f) | 421 | 94% | 4% | 0% | 1% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | mid(8-20) | 114 | 83% | 0% | 15% | 0% | 2% |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | small/crop512/fp16/box | 50 | ALL | 550 | 92% | 3% | 3% | 1% | 1% |
| astonvilla_seg080 | small/full/fp16/box | 10 | slow(<8px/f) | 387 | 97% | 1% | 0% | 2% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 10 | mid(8-20) | 103 | 97% | 2% | 1% | 0% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 10 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 10 | ALL | 505 | 97% | 1% | 0% | 1% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 25 | slow(<8px/f) | 413 | 95% | 2% | 1% | 2% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 25 | mid(8-20) | 111 | 91% | 4% | 5% | 0% | 1% |
| astonvilla_seg080 | small/full/fp16/box | 25 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 25 | ALL | 539 | 94% | 2% | 2% | 1% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 50 | slow(<8px/f) | 421 | 94% | 3% | 1% | 1% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 50 | mid(8-20) | 114 | 82% | 3% | 14% | 0% | 2% |
| astonvilla_seg080 | small/full/fp16/box | 50 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | small/full/fp16/box | 50 | ALL | 550 | 91% | 3% | 4% | 1% | 1% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 10 | slow(<8px/f) | 387 | 98% | 0% | 0% | 2% | 1% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 10 | mid(8-20) | 103 | 96% | 0% | 1% | 0% | 3% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 10 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 10 | ALL | 505 | 97% | 0% | 0% | 1% | 1% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 25 | slow(<8px/f) | 413 | 96% | 2% | 0% | 1% | 1% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 25 | mid(8-20) | 111 | 88% | 1% | 10% | 0% | 1% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 25 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 25 | ALL | 539 | 94% | 2% | 2% | 1% | 1% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | slow(<8px/f) | 421 | 95% | 3% | 0% | 1% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | mid(8-20) | 114 | 84% | 0% | 15% | 0% | 1% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | tiny/crop512/fp16/box | 50 | ALL | 550 | 92% | 3% | 4% | 1% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 10 | slow(<8px/f) | 387 | 98% | 0% | 0% | 2% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 10 | mid(8-20) | 103 | 94% | 4% | 2% | 0% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 10 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 10 | ALL | 505 | 97% | 1% | 1% | 1% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 25 | slow(<8px/f) | 413 | 96% | 2% | 0% | 1% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 25 | mid(8-20) | 111 | 93% | 4% | 4% | 0% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 25 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 25 | ALL | 539 | 95% | 3% | 1% | 1% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | slow(<8px/f) | 421 | 94% | 4% | 1% | 1% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | mid(8-20) | 114 | 93% | 4% | 4% | 0% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | fast(>20) | 15 | 93% | 0% | 7% | 0% | 0% |
| astonvilla_seg080 | tiny/full/fp16/box | 50 | ALL | 550 | 93% | 4% | 2% | 1% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 10 | slow(<8px/f) | 170 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 10 | mid(8-20) | 25 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 10 | ALL | 195 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 25 | slow(<8px/f) | 181 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 25 | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 25 | ALL | 208 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 50 | slow(<8px/f) | 185 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 50 | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 50 | ALL | 212 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 10 | slow(<8px/f) | 170 | 99% | 1% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 10 | mid(8-20) | 25 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 10 | ALL | 195 | 99% | 1% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 25 | slow(<8px/f) | 181 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 25 | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 25 | ALL | 208 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 50 | slow(<8px/f) | 185 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 50 | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 50 | ALL | 212 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 10 | slow(<8px/f) | 170 | 99% | 1% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 10 | mid(8-20) | 25 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 10 | ALL | 195 | 99% | 1% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 25 | slow(<8px/f) | 181 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 25 | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 25 | ALL | 208 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 50 | slow(<8px/f) | 185 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 50 | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 50 | ALL | 212 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 10 | slow(<8px/f) | 170 | 99% | 1% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 10 | mid(8-20) | 25 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 10 | ALL | 195 | 99% | 1% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 25 | slow(<8px/f) | 181 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 25 | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 25 | ALL | 208 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 50 | slow(<8px/f) | 185 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 50 | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | small/full/fp16/box | 50 | ALL | 212 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 10 | slow(<8px/f) | 170 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 10 | mid(8-20) | 25 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 10 | ALL | 195 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 25 | slow(<8px/f) | 181 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 25 | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 25 | ALL | 208 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 50 | slow(<8px/f) | 185 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 50 | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 50 | ALL | 212 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 10 | slow(<8px/f) | 170 | 99% | 1% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 10 | mid(8-20) | 25 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 10 | ALL | 195 | 99% | 1% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 25 | slow(<8px/f) | 181 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 25 | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 25 | ALL | 208 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 50 | slow(<8px/f) | 185 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 50 | mid(8-20) | 27 | 100% | 0% | 0% | 0% | 0% |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 50 | ALL | 212 | 100% | 0% | 0% | 0% | 0% |

### Runtime and memory

| clip | config | frames propagated | s/frame | peak VRAM alloc MB | peak VRAM reserved MB | GPU |
|---|---|---|---|---|---|---|
| arsenal_example | base_plus/crop512/fp16/box | 757 | 0.1665 | 957 | 1170 | NVIDIA GeForce RTX 2070 |
| arsenal_example | base_plus/full/fp16/box | 757 | 0.1771 | 963 | 1154 | NVIDIA GeForce RTX 2070 |
| arsenal_example | small/crop512/fp16/box | 757 | 0.1372 | 325 | 656 | NVIDIA GeForce RTX 2070 |
| arsenal_example | small/full/fp16/box | 757 | 0.1544 | 315 | 328 | NVIDIA GeForce RTX 2070 |
| arsenal_example | tiny/crop512/fp16/box | 757 | 0.131 | 688 | 878 | NVIDIA GeForce RTX 2070 |
| arsenal_example | tiny/full/fp16/box | 757 | 0.1493 | 694 | 846 | NVIDIA GeForce RTX 2070 |
| astonvilla_seg080 | base_plus/crop512/fp16/box | 3900 | 0.1651 | 957 | 1172 | NVIDIA GeForce RTX 2070 |
| astonvilla_seg080 | base_plus/full/fp16/box | 3900 | 0.1724 | 963 | 1154 | NVIDIA GeForce RTX 2070 |
| astonvilla_seg080 | small/crop512/fp16/box | 3900 | 0.1555 | 731 | 946 | NVIDIA GeForce RTX 2070 |
| astonvilla_seg080 | small/full/fp16/box | 3900 | 0.1671 | 730 | 860 | NVIDIA GeForce RTX 2070 |
| astonvilla_seg080 | tiny/crop512/fp16/box | 3900 | 0.1323 | 689 | 878 | NVIDIA GeForce RTX 2070 |
| astonvilla_seg080 | tiny/full/fp16/box | 3900 | 0.1341 | 694 | 846 | NVIDIA GeForce RTX 2070 |
| bournemouth_1st_seg070 | base_plus/crop512/fp16/box | 1508 | 0.1645 | 957 | 1172 | NVIDIA GeForce RTX 2070 |
| bournemouth_1st_seg070 | base_plus/full/fp16/box | 1508 | 0.1739 | 963 | 1154 | NVIDIA GeForce RTX 2070 |
| bournemouth_1st_seg070 | small/crop512/fp16/box | 1508 | 0.1385 | 731 | 946 | NVIDIA GeForce RTX 2070 |
| bournemouth_1st_seg070 | small/full/fp16/box | 1508 | 0.1547 | 734 | 860 | NVIDIA GeForce RTX 2070 |
| bournemouth_1st_seg070 | tiny/crop512/fp16/box | 1508 | 0.1323 | 689 | 878 | NVIDIA GeForce RTX 2070 |
| bournemouth_1st_seg070 | tiny/full/fp16/box | 1508 | 0.1406 | 695 | 846 | NVIDIA GeForce RTX 2070 |
