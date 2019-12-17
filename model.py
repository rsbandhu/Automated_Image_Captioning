import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderRNN, self).__init__()
        #print("inside init of class DecoderRNN")
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers = num_layers, batch_first = True)        
        self.hidden2Tag = nn.Linear(hidden_size, vocab_size)
                
            
    def forward(self, features, captions):
        embeds = self.word_embeddings(captions)
        features_unsq = features.unsqueeze(1)
        
        #print(" types **##   ", embeds.type(), features_unsq.type())
        #print("Embed shape and unsqueeze features shape = ***** ", embeds.shape, features_unsq.shape)
        
        #Cocatenate features extracted from the image followed by caption
        features_caption_cat = torch.cat((features_unsq, embeds), 1)
        
        #remove the "end" word from the end
        features_caption_cat = features_caption_cat[:, :-1,:]
        
        lstm_out, (ht,ct) = self.lstm(features_caption_cat)
        
        decoder_out = self.hidden2Tag(lstm_out)
        
        return decoder_out
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        caption_prediction = []  # this array will contain the indices of the predcited words in caption in order
        
        for i in range(max_len):
            inputs, states = self.lstm(inputs, states) 
            decoder_out = self.hidden2Tag(inputs)  # output of the linear unit after hidden state
            
            val, max_idx = torch.max(decoder_out, 2)
            word_idx = int(max_idx) #index of the word with max prob
            
            #print("max index = ", word_idx)
            caption_prediction.append(word_idx)
            
            inputs = self.word_embeddings(max_idx)  # generate the embedding for the word predicted in this step   
          
        return caption_prediction
  