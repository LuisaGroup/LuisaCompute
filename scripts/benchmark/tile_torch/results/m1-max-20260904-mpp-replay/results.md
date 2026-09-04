# Native MPP versus MPS

FP32 C=A×B, no transpose/prepacking, private compact row-major buffers, alpha=1, beta=0. MPP relaxed precision and compiler fast math are disabled. Complete outputs use the same FP64 oracle (atol=rtol=1e-4).

Host wall time includes encoding, submission and synchronization. GPU time is the command-buffer/batch interval, including dispatch and synchronization costs; it is not an isolated instruction latency. Setup, uploads and validation are untimed.

Frozen replay: each shape visits all six orders of MPS/default-MPP/selected-MPP. Values are medians of per-round medians; ratios are paired round medians [min,max], not confidence intervals. Ratio >1 means selected MPP is slower. No failed/slow round is removed.

| M×N×K | Rounds | MPS GPU µs | Default MPP GPU µs | Selected MPP GPU µs | MPP/MPS GPU | Host MPP/MPS |
|---|---:|---:|---:|---:|---:|---:|
| 32x32x32 | 6 | 8.592 | 6.129 | 2.399 | 0.280 [0.270,0.288] | 0.256 |
| 128x128x128 | 6 | 13.057 | 10.337 | 4.597 | 0.347 [0.339,0.354] | 0.325 |
| 512x512x512 | 6 | 50.359 | 48.379 | 45.802 | 0.920 [0.891,0.929] | 0.900 |
| 1024x1024x1024 | 6 | 275.036 | 342.627 | 267.788 | 0.975 [0.917,1.000] | 0.969 |
| 256x1024x128 | 6 | 17.876 | 16.652 | 16.359 | 0.916 [0.890,0.978] | 0.867 |
| 1024x128x256 | 6 | 23.493 | 19.666 | 17.422 | 0.741 [0.730,0.748] | 0.711 |
| 127x193x61 | 6 | 14.991 | 11.305 | 6.692 | 0.449 [0.427,0.462] | 0.422 |
| 513x257x129 | 6 | 34.024 | 27.184 | 24.298 | 0.714 [0.706,0.725] | 0.694 |
| 2048x2048x2048 | 6 | 2189.644 | 3052.561 | 2111.258 | 0.975 [0.934,1.040] | 0.967 |
| 256x2048x1024 | 6 | 139.257 | 187.737 | 151.648 | 1.088 [1.063,1.092] | 1.082 |
| 2048x256x1024 | 6 | 140.842 | 185.828 | 150.773 | 1.074 [1.039,1.112] | 1.065 |

Artifacts unchanged: True.

Raw samples, errors, configurations and artifact hashes: [results.json](results.json).
