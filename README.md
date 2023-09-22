## Implementation of *TaxoGAN*, ICDM 2020.

Please cite the following work if you find the code useful.

```
@inproceedings{yang2020taxogan,
	Author = {Yang, Carl and Zhang, Jieyu and Han, Jiawei},
	Booktitle = {ICDM},
	Title = {Co-Embedding Network Nodes and Hierarchical Labels with Taxonomy Based Generative Adversarial Networks},
	Year = {2020}
}
```
Contact: Jieyu Zhang (jieyuz2@illinois.edu), Carl Yang (yangji9181@gmail.com)


## Prerequisites
- Python3
- Pytorch 1.4

## Training 
```
python3 src/main.py --gpu 0 --dataset dblp --model TaxoGAN_V3 --task taxonomy --early_stop 0 --transform 1
```
