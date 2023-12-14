# SeqHiC2RNA

### Attention
* When changing the model's feature dimension, we need to change the dim_divisible_by in config simultaneously!! I may think of a better way to deal with this in code later
* Currently only deal with the situation that output head number is 1, if we need to increase the number, we need to change function evalutation()
* Actually current implementation still use the enformer-package for the function str_to_one_hot, seq_indices_to_one_hot, maybe we can fix it later. It will not bother us currently.
* I use the formula (ascii code + 1) % 5 to convert the ATGCN -> 10234 in order to meet the standard in enformer-pytorch package that 4 will be convert to [0, 0, 0, 0]
* I repeat the target gene expression 8 times to convert the 1024-resolution result to 128-resolution result
* Maybe we need to change the dimension of the enformer when using our data, currently I still use the same dimension(1536) with the enformer paper.
* I haven't tune the dropout rate, maybe we can try it later. 

### notes
* Many different file in algo/ is just for testing the previous saved model, the newest algorithm is in Hcformer_pretrain.py