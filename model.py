import torch.nn as nn
import torch
from module import Linear
import math
from transformers import BertModel
import torch.utils.checkpoint as checkpoint

class Classifier(nn.Module):
    def __init__(self, config, num_intent, num_slot):
        super(Classifier, self).__init__()
        self.bert_dim = config.bert_dim
        self.checkpoint_batch_size = config.checkpoint_batch_size
        self.num_intent = num_intent
        self.num_slot = num_slot
        self.sent_encoder = BertModel.from_pretrained('bert-base-multilingual-cased')

        self.mlp_intent = Linear(self.bert_dim, self.num_intent, bias = True)
        self.mlp_slot = Linear(self.bert_dim, self.num_slot, bias = True)

        self.slot_norm = nn.LayerNorm(self.bert_dim)
        self.intent_norm = nn.LayerNorm(self.bert_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
            
    def embed_sentences_checkpointed(self, input_ids, attention_mask, token_type_ids, checkpoint_batch_size=-1):
        if checkpoint_batch_size < 0 or input_ids.shape[0] < checkpoint_batch_size:
            checkpoint_batch_size = input_ids.shape[0]
        
        # prepare implicit variables
        device = input_ids.device
        input_shape = input_ids.size()
        head_mask = [None] * self.sent_encoder.config.num_hidden_layers
        extended_attention_mask:torch.Tensor = self.sent_encoder.get_extended_attention_mask(attention_mask, input_shape, device)

        # define function for checkpointing
        def partial_encode(*inputs):
            encoder_outputs = self.sent_encoder.encoder(inputs[0], attention_mask=inputs[1], head_mask=head_mask,)
            pooled_output = encoder_outputs[0]
            return pooled_output

        # run embedding layer on everything at once
        embedding_output = self.sent_encoder.embeddings(input_ids=input_ids, position_ids=None,
                                                        token_type_ids=token_type_ids, inputs_embeds=None)
        
        # run encoding and pooling on one mini-batch at a time
        pooled_output_list = []
        for b in range(math.ceil(input_ids.shape[0] / checkpoint_batch_size)):
            b_embedding_output = embedding_output[b * checkpoint_batch_size : (b + 1) * checkpoint_batch_size]
            b_attention_mask = extended_attention_mask[b * checkpoint_batch_size : (b + 1) * checkpoint_batch_size]
            pooled_output = checkpoint.checkpoint(partial_encode, b_embedding_output, b_attention_mask)
            pooled_output_list.append(pooled_output)
        pooled_output_list = torch.cat(pooled_output_list, dim=0)
        return pooled_output_list
    
    def forward(self, x_ids, x_mask, x_type_ids, label_intent = None, label_slot = None):
        x_embed = self.embed_sentences_checkpointed(x_ids, x_mask, x_type_ids, self.checkpoint_batch_size)
        sent_embed = self.sent_encoder.pooler(x_embed)
        
        score_intent = self.mlp_intent(sent_embed)
        score_slot = self.mlp_slot(x_embed)
        
        score_slot = (score_slot * x_mask.unsqueeze(-1)).reshape(-1, self.num_slot)
        loss = None
        if label_intent is not None:
            loss_intent = self.ce_loss(score_intent, label_intent)
            loss_slot = self.ce_loss(score_slot, label_slot)
            loss = loss_intent + loss_slot
            
        return loss, score_intent, score_slot