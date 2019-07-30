import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, input_size, settings):
        super(CNN, self).__init__()

        self.output_size = settings.nb_classes
        self.hidden_size = settings.hidden_dim
        last_input_size = settings.hidden_dim
        self.kernel_size = 3

        self.conv1 = nn.Conv1d(input_size, self.hidden_size, kernel_size=self.kernel_size,
                               stride=1, padding=(self.kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(self.hidden_size, last_input_size,
                               kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2)
        self.conv3 = nn.Conv1d(self.hidden_size, last_input_size,
                               kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2)
        self.conv4 = nn.Conv1d(self.hidden_size, last_input_size,
                               kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2)

        self.output_layer = torch.nn.Linear(last_input_size, self.output_size)

    def forward(self, x, mean_field_inference=False):
        """CNN forward pass
        
        Mean: last layer, mean on sequence
            - take packed output from last layer (out) that contains all time steps for the last layer
           - find where padding was done and create a mask for those values, apply this mask
            - take a mean for the whole sequence (time_steps)
            - use h2o to obtain output (beware! it is only one layer deep since it is the last one only)
        
        Arguments:
            x {Packed tensor} -- 
        
        Keyword Arguments:
            mean_field_inference {bool} -- [description] (default: {False})

        """

        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, lens = torch.nn.utils.rnn.pad_packed_sequence(x)
            # x is (seq_len, batch, hidden size)
            # change to (B,H,L) for CNN
            x = x.permute(1,2,0).to(x.device)
            # create the mask
            # using broadcasting
            lens = lens.unsqueeze(1) # shape (B,1)
            tmp = torch.arange(x.size(2)).unsqueeze(0)  # size (1,L)
            mask = tmp.float() < lens.float() # (B,L)
            mask = mask.unsqueeze(1).to(x.device) # (B,1,L)
        else:
            # only permute
            x = x.permute(1,2,0).to(x.device)
            mask = torch.ones(x.size(0)).unsqueeze(1).unsqueeze(1).to(x.device)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # import ipdb; ipdb.set_trace()

        # masking padded values
        x = mask.float() * x
        
        # mean pooling
        x = x.sum(2) / mask.float().sum(2)
        output = self.output_layer(x)

        return output
