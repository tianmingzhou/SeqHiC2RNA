# SeqHiC2RNA

### Attention
* Actually current implementation still use the enformer-package in algo/enformer.py in line 10 and 12, maybe we can fix it later. It will not bother us currently.
* I use the formula (ascii code + 1) % 5 to convert the ATGCN -> 10234 in order to meet the standard in enformer-pytorch package that 4 will be convert to [0, 0, 0, 0]
* I repeat the target gene expression 8 times to convert the 1024-resolution result to 128-resolution result
* Maybe we need to change the dimension of the enformer when using our data, currently I still use the same dimension(1536) with the enformer paper.
* I haven't tune the dropout rate, maybe we can try it later. 