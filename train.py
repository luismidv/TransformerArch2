import torch
from torch import nn
import torch.optim

import dataprepare
import transformerarch


#VOCAB PARAMETERS
src_vocab_size = 5000
tgt_vocab_size = 5000
max_seq_length = 100
num_heads = 8
model_dimension = 512
d_ff = 2048
dropout = 0.2


#FEATURES AND LABELS
features = torch.randint(1,src_vocab_size, (64, max_seq_length))
labels = torch.randint(1,src_vocab_size, (64, max_seq_length))

#DATAPREPARE LOAD
features_load, labels_load = dataprepare.csv_loader('./data/data.csv')


#MASK CREATION
src_mask, tgt_mask = transformerarch.generate_mask(features, labels)

#DROPOUT DEFINITION
dropout_value = 0.1
dropout = nn.Dropout(dropout_value)

#CREATE ENCODER EMBEDDINGS AND EMBED TRAINING EXAMPLES


encoder_embe = nn.Embedding(src_vocab_size,model_dimension)
feature_embedded = encoder_embe(features)
label_embedded = encoder_embe(labels)

#COMPUTE POSITIONAL ENCODING AND DROPOUTS

pos_enc = transformerarch.PositionEncoding(model_dimension, max_seq_length)
feature_embed = dropout(pos_enc(feature_embedded))
label_embed = dropout(pos_enc(label_embedded))

#CREATE TRANSFORMER LOSS FN AND OPTIMIZER
transfomer_arch = transformerarch.Transformer(model_dimension,max_seq_length,d_ff,dropout_value,src_mask,tgt_vocab_size, num_heads)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transfomer_arch.parameters(), lr= 1e-4)

epochs = 100
transfomer_arch.train()

for epoch in range(epochs):
    #CLEAN GRADS FROM LAST RUN
    optimizer.zero_grad()

    #COMPUTE TRANSFORMER OUTPUT
    trf_output = transfomer_arch(feature_embed,label_embed, src_mask,tgt_mask)

    #RESHAPING
    distribution = trf_output.contiguous().view(-1, tgt_vocab_size)
    new_labels = labels.contiguous().view(-1)

    loss = loss_fn(distribution, new_labels)
    print(f"Current model Loss {loss.item() *100:.2f}")
    loss.backward(retain_graph=True)
    optimizer.step()
