import torch
import torch.nn as nn
import esm
import esm.inverse_folding

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()

        # Load IF-model
        self.if_model, self.if_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

        # Load ESM-2 model
        self.seq_model, self.seq_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.seq_alphabet.get_batch_converter()

        # LayerNorm applied to output of sequence and structure embeddings
        d_seq = 1280
        d_struct = 512
        self.norm_seq = nn.LayerNorm(d_seq)
        self.norm_struct = nn.LayerNorm(d_struct)

        # to do -- add more layers based on model hyperparameter
        self.linear_in = nn.Linear(d_seq + d_struct, 1)
        self.final_act = nn.Sigmoid()


    def forward(self, x):
        batch_coords, batch_tokens = x

        seq_emb = self.seq_model(
            batch_tokens, repr_layers=[33], return_contacts=True)["representations"][33]
        # drop the first and last sequence embedding since these are for start and end tokens
        seq_emb = seq_emb[:, 1:seq_emb.shape[1]-1, :]

        # will need to use padding to maintain same shape if there are multiple proteins in batch
        struct_emb = torch.stack([
            esm.inverse_folding.util.get_encoder_output(
                self.if_model, self.if_alphabet, coords)
            for coords in batch_coords
        ])
        
        print(seq_emb.shape)
        print(struct_emb.shape)

        seq_emb = self.norm_seq(seq_emb)
        struct_emb = self.norm_struct(struct_emb)

        print(seq_emb.shape)
        print(struct_emb.shape)

        feat = torch.cat((seq_emb, struct_emb), dim=-1)
        output = self.final_act(self.linear_in(feat))
        return output


model = FusionModel()

structure = esm.inverse_folding.util.load_structure('examples/inverse_folding/data/5YH2.pdb', 'A')
coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)

data = [
    ("5YH2", seq),
]
batch_labels, batch_strs, batch_tokens = model.batch_converter(data)
# batch_lens = (batch_tokens != model.seq_alphabet.padding_idx).sum(1)

pred = model(([coords], batch_tokens))

print(pred.shape)

# now test backprop
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

y_true = torch.rand(pred.shape)

loss = loss_fn(pred, y_true)
print(loss)

optimizer.zero_grad()
loss.backward()
optimizer.step()

pred = model(([coords], batch_tokens))
loss = loss_fn(pred, y_true)
print(loss)