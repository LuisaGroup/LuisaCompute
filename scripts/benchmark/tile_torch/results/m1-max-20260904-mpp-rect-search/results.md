# Native MPP versus MPS

FP32 C=A×B, no transpose/prepacking, private compact row-major buffers, alpha=1, beta=0. MPP relaxed precision and compiler fast math are disabled. Complete outputs use the same FP64 oracle (atol=rtol=1e-4).

Host wall time includes encoding, submission and synchronization. GPU time is the command-buffer/batch interval, including dispatch and synchronization costs; it is not an isolated instruction latency. Setup, uploads and validation are untimed.

Exploratory search only. Selected minima require a separate frozen, counterbalanced replay.

| M×N×K | Config (M,N,op-SG,coop,static-K,inline,group-SG,cohort-M) | Valid | Host µs | GPU µs |
|---|---|---|---:|---:|
| 256x2048x1024 | MPS | yes | 185.196 | 179.171 |
| 256x2048x1024 | 32,32,1,1,0,1,1,1 | yes | 196.029 | 192.959 |
| 256x2048x1024 | 32,64,1,1,0,1,1,1 | yes | 230.343 | 224.178 |
| 256x2048x1024 | 64,32,1,1,0,1,1,1 | yes | 243.221 | 237.213 |
| 256x2048x1024 | 32,64,1,1,0,1,4,1 | yes | 227.041 | 224.856 |
| 256x2048x1024 | 64,32,1,1,0,1,4,4 | yes | 243.843 | 241.401 |
| 256x2048x1024 | 32,64,2,1,0,1,2,1 | yes | 233.131 | 230.525 |
| 256x2048x1024 | 64,32,2,1,0,1,2,1 | yes | 191.106 | 189.118 |
| 256x2048x1024 | 32,32,2,1,0,1,2,1 | yes | 222.732 | 217.476 |
| 256x2048x1024 | 32,32,4,1,0,1,4,1 | yes | 388.976 | 380.168 |
| 256x2048x1024 | 32,64,4,1,0,1,4,1 | yes | 324.982 | 319.292 |
| 256x2048x1024 | 64,32,4,1,0,1,4,1 | yes | 271.154 | 268.069 |
| 256x2048x1024 | 48,32,1,1,0,1,1,1 | yes | 254.051 | 251.230 |
| 256x2048x1024 | 32,48,1,1,0,1,1,1 | yes | 259.409 | 255.138 |
| 256x2048x1024 | 16,128,2,1,0,1,2,1 | yes | 295.753 | 289.109 |
| 256x2048x1024 | 128,16,2,1,0,1,2,1 | yes | 306.876 | 302.052 |
| 2048x256x1024 | MPS | yes | 176.120 | 171.737 |
| 2048x256x1024 | 32,64,1,1,0,1,1,1 | yes | 237.138 | 232.303 |
| 2048x256x1024 | 64,32,1,1,0,1,1,1 | yes | 273.889 | 269.821 |
| 2048x256x1024 | 32,64,1,1,0,1,4,1 | yes | 262.703 | 259.999 |
| 2048x256x1024 | 64,32,1,1,0,1,4,4 | yes | 250.540 | 246.313 |
| 2048x256x1024 | 32,64,2,1,0,1,2,1 | yes | 224.815 | 221.722 |
| 2048x256x1024 | 64,32,2,1,0,1,2,1 | yes | 195.473 | 192.365 |
| 2048x256x1024 | 32,32,2,1,0,1,2,1 | yes | 217.314 | 214.725 |
| 2048x256x1024 | 32,32,4,1,0,1,4,1 | yes | 377.020 | 373.354 |
| 2048x256x1024 | 32,64,4,1,0,1,4,1 | yes | 324.661 | 319.894 |
| 2048x256x1024 | 64,32,4,1,0,1,4,1 | yes | 244.240 | 237.622 |
| 2048x256x1024 | 48,32,1,1,0,1,1,1 | yes | 218.454 | 216.045 |
| 2048x256x1024 | 32,48,1,1,0,1,1,1 | yes | 218.974 | 216.640 |
| 2048x256x1024 | 16,128,2,1,0,1,2,1 | yes | 272.274 | 268.703 |
| 2048x256x1024 | 128,16,2,1,0,1,2,1 | yes | 297.504 | 294.455 |
| 2048x256x1024 | 32,32,1,1,0,1,1,1 | yes | 163.937 | 161.605 |

Artifacts unchanged: True.

Raw samples, errors, configurations and artifact hashes: [results.json](results.json).
