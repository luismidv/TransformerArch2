import torch
import torch.nn as nn
import torch.optim
import math

from sympy.physics.units import second


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dimension, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        #PARAMETERS
        self.model_dimension = model_dimension
        self.num_heads = num_heads
        self.d_k = model_dimension // num_heads

        #LINEAR TRANSFORMATIONS
        self.w_query = nn.Linear(model_dimension,model_dimension)
        self.w_value = nn.Linear(model_dimension, model_dimension)
        self.w_key = nn.Linear(model_dimension, model_dimension)
        self.w_o = nn.Linear(model_dimension, model_dimension)

        #scale_dotproduct_attention()

    def split_heads(self,x):
        print(f"Feature size {x.size()}")
        batch_size, seq_length, model_dimension = x.size()
        return x.view(batch_size,seq_length,self.num_heads, self.d_k).transpose(1,2)
    
    def scale_dotproduct_attention(self, query, key, values, mask = None):
        dot_product = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
             dot_product = dot_product.masked_fill(mask == 0, -1e9)
        atention_probs = torch.softmax(dot_product, dim = 1)
        output = torch.matmul(atention_probs, values)
        return output

    def combine_heads(self,x):
         batch_size,_,seq_length,d_k = x.size()
         return x.transpose(1,2).contiguous().view(batch_size, seq_length,self.model_dimension)
          


    def forward(self,query,key,value,mask = None):
        #LINEAR TRANSFORMATION AND SPLIT INTO HEADS
        print(f"Tamaño de la query {query.shape}")
        query = self.split_heads(self.w_query(query))
        key = self.split_heads(self.w_key(key))
        value = self.split_heads(self.w_value(value))



        atention_output = self.scale_dotproduct_attention(query,key,value,mask)
        print(f"Atention output {atention_output.shape}")
        atention_combined = self.combine_heads(atention_output)
        print(f"Atention output {atention_combined.shape}")
        output = self.w_o(atention_combined)
        return output 
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self,model_dimension, d_feedforw):
        super(PositionWiseFeedForward,self).__init__()
        self.linear1 = nn.Linear(model_dimension, d_feedforw)
        self.linear2 = nn.Linear(d_feedforw,model_dimension)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class PositionEncoding(nn.Module):
    def __init__(self,model_dimension, max_seq_length):
        super(PositionEncoding,self).__init__()
        pe = torch.zeros(max_seq_length, model_dimension)
        position = torch.arange(0, max_seq_length, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dimension, 2).float() * -(math.log(10000.0))/model_dimension)    
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self,x):
        return x + self.pe[:, :x.size(1)]

class Encoder(nn.Module):
     
    def __init__(self, model_dimension, num_heads, d_ff, max_seq_length,dropout,mask):
        super(Encoder, self).__init__()
        self.multiH = MultiHeadAttention(model_dimension, num_heads)
        self.possw = PositionWiseFeedForward(model_dimension, d_ff)
        self.posenc = PositionEncoding(model_dimension, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dimension)
        self.norm2 = nn.LayerNorm(model_dimension)
    
    def forward(self,x,mask):
        attention = self.multiH(x,x,x,mask)
        normalized = self.norm1(x + self.dropout(attention))
        posswise = self.possw(normalized)
        output = self.norm2(normalized + self.dropout(posswise))
        return output
    
class Decoder(nn.Module):
    def __init__(self,model_dimension, num_heads,d_ff,max_seq_length,dropout,mask ):
        super(Decoder, self).__init__()
        self.multiH = MultiHeadAttention(model_dimension, num_heads)
        self.crossH = MultiHeadAttention(model_dimension, num_heads)

        self.possw = PositionWiseFeedForward(model_dimension, d_ff)
        self.possenc = PositionEncoding(model_dimension, max_seq_length)

        self.layer1 = nn.LayerNorm(model_dimension)
        self.layer2 = nn.LayerNorm(model_dimension)
        self.layer3 = nn.LayerNorm(model_dimension)
        self.layerdropout = nn.Dropout(dropout)
    
    def forward(self,x, enc_output,src_mask):
        attention = self.multiH(x,x,x,src_mask)
        first_norm = self.layer1(x + self.layerdropout(attention))
        second_attention = self.crossH(first_norm, enc_output, enc_output, src_mask)
        second_norm = self.layer2(first_norm + self.layerdropout(second_attention))
        forw_output = self.possw(second_norm)
        output = self.layer3(second_norm + self.layerdropout(forw_output))

        return output
    

def generate_mask(src,tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal = 1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

class Transformer(nn.Module):
    def __init__(self, model_dimension, max_seq_length, d_ff, dropout, mask):
        super(Transformer, self).__init__()
        self.poss_enc = PositionEncoding(model_dimension, max_seq_length)
        self.encoder = Encoder(model_dimension, num_heads, d_ff, max_seq_length, dropout, mask)
        self.decoder = Decoder(model_dimension, num_heads, d_ff, max_seq_length, dropout ,mask)

    def forward(self, features,mask):
        #TODO FIX POSITIONAL ENCODING, GETTING NOT A NUMBER MATRIX IF USING IT
        encoded_feature = self.poss_enc(features)

        encoder_output = self.encoder(encoded_feature,mask)
        print(f"Encoder output matrix {encoder_output}")
        decoder_output = self.decoder(encoded_feature,encoder_output, mask)

        print(f"Decoder output matrix {decoder_output}")

        return decoder_output


#VOCAB PARAMETERS
src_vocab_size = 5000
tgt_vocab_size = 5000

max_seq_length = 100
num_heads = 8

model_dimension = 512
d_ff = 2048
dropout = 0.2


#FEATURES AND LABELS

features = torch.randint(1,src_vocab_size, (64, max_seq_length), dtype=torch.int)
labels = torch.randint(1,src_vocab_size, (64, max_seq_length),   dtype = torch.int)

src_mask, tgt_mask = generate_mask(features, labels)

multiH = MultiHeadAttention(model_dimension, 8)
posenc = PositionEncoding(model_dimension, max_seq_length)
dropout1 = nn.Dropout(0.1)

encoder_embe = nn.Embedding(src_vocab_size,model_dimension)
feature_embedded = encoder_embe(features)
src_embed = dropout1(feature_embedded)

encoder = Encoder(model_dimension, num_heads, d_ff, max_seq_length, dropout, src_mask)
decoder = Decoder(model_dimension, num_heads, d_ff, max_seq_length, dropout ,src_mask)

encoder_output = encoder(src_embed,src_mask)
decoder_output = decoder(src_embed,encoder_output, src_mask)

transfomer_arch = Transformer(model_dimension,max_seq_length,d_ff,dropout,src_mask)
trf_output = transfomer_arch(src_embed,src_mask)




#encoder = Encoder(model_dimension, num_heads, d_ff, max_seq_length,dropout,src_mask)
#decoder = Decoder(model_dimension, num_heads, d_ff, max_seq_length,dropout,src_mask)
#output = multiH(src_embed,src_embed,src_embed)
#encoder_output = encoder(feature_embedded,src_mask)
#decoder_output = decoder(feature_embedded,tgt_mask,encoder_output,src_mask)


