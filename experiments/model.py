import torch

class TNet(torch.nn.Module):
    def __init__(self,
            in_dim: int,
            out_dim: int,
            up_k: int = 10,
            up_l: int = 30,
            down_k: int = 60,
            down_l: int = 10,
            device: str = 'cuda',
            tau: float = 1.,
            descent_layer : bool = False,
            descent_layer_in : bool = False):
        super().__init__()
        
        self.descent_layer = descent_layer
        self.descent_layer_in = descent_layer_in

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.up_layer_node = up_k
        self.up_layer_num = up_l
        self.down_layer_node = down_k
        self.down_layer_num = down_l

        self.device = device
        self.tau = tau
        self.layer_attention = False
        
        self.layer_nodenum_list = [self.in_dim]+[self.down_layer_node for _ in range(self.down_layer_num)]+[self.up_layer_node for _ in range(self.up_layer_num)]

        # layers
        self.layer_flatten = torch.nn.Flatten()

        self.layer_list = self.layer_nodenum_list + [self.out_dim]
        self.layers = self.generate_layer(self.layer_list)

        # node num of layers
        self.layer_sum_list = self.layer_list.copy()
        for i in range(len(self.layer_list)):
            self.layer_sum_list[i] = sum(self.layer_list[:i+1])
        self.groups = torch.tensor([0]+self.layer_sum_list)
        
    def generate_layer(self, layer_list):
        layers = torch.nn.ParameterList([])
        self.which_layers = torch.nn.ParameterList([])
        for num in range(len(layer_list)-1):
            layers.append(torch.nn.parameter.Parameter(torch.randn(2, layer_list[num+1], sum(layer_list[0:num+1]), 2, device=self.device)))
            self.which_layers.append(torch.nn.parameter.Parameter(torch.randn(2, layer_list[num+1], num+1, device=self.device)))
        return layers

    def forward(self, x):
        x = self.layer_flatten(x)
        x_in = x
        if self.training:
            for i in range(len(self.layers)):
                ab = self.layers[i] 
                which_layer = self.which_layers[i] # 2 * out * sum(in) * 2
                
                input_layer = torch.abs(ab[:,:,:self.layer_list[0],:]).flatten(start_dim=-2,end_dim=-1)
                input_layer = torch.nn.functional.gumbel_softmax(input_layer, tau=self.tau, hard=False).view(2, -1, self.layer_list[0], 2) # 2 * out * sum(in) * 2
                
                down_body_layer = torch.abs(ab[:,:,self.layer_sum_list[0]:self.layer_sum_list[self.down_layer_num],:]) # 2*out*sum(down_layer)*2
                down_body_layer = down_body_layer.view(2, ab.size(1), -1, self.down_layer_node, 2)
                down_body_layer = torch.nn.functional.gumbel_softmax(down_body_layer.flatten(start_dim=-2,end_dim=-1), tau=self.tau, hard=False) #2*out*down_layer*(down_node*2)
                down_body_layer = down_body_layer.view(2, ab.size(1),-1,2) #2*out*(down_layer*down_node)
                
                up_body_layer = torch.abs(ab[:,:,self.layer_sum_list[self.down_layer_num]:,:])
                up_body_layer = up_body_layer.view(2, ab.size(1), -1, self.up_layer_node, 2) # 2 * out * sum(up) * 2
                up_body_layer = torch.nn.functional.gumbel_softmax(up_body_layer.flatten(start_dim=-2,end_dim=-1), tau=self.tau, hard=False)
                up_body_layer = up_body_layer.view(2, ab.size(1),-1,2)
                
                ab = torch.cat([input_layer, down_body_layer, up_body_layer], dim = -2)

                a_unit = (x_in.unsqueeze(2) * ab[0,:,:,0].t().unsqueeze(0) + ab[0,:,:,1].t().unsqueeze(0) * (1-x_in.unsqueeze(2))).transpose(dim0=-1, dim1=-2)
                b_unit = (x_in.unsqueeze(2) * ab[1,:,:,0].t().unsqueeze(0) + (1-x_in.unsqueeze(2)) * ab[1,:,:,1].t().unsqueeze(0)).transpose(dim0=-1, dim1=-2)

                groups_ = self.layer_list[:i+1]

                if self.descent_layer > 0 and i > 4 :
                    # regularized_skip_connection weight
                    decent = torch.linspace(0.1, 1, which_layer.size(-1), device='cuda')
                    which_layer = torch.abs(which_layer) * decent
                elif self.descent_layer_in and 0 < i <= 4:
                    decent = torch.linspace(1, 0.1, which_layer.size(-1), device='cuda')
                    which_layer = torch.abs(which_layer) * decent

                which_layer = torch.nn.functional.gumbel_softmax(which_layer, tau=self.tau, hard=False) 
                which_layer_ = torch.repeat_interleave(which_layer, torch.tensor(groups_ ,device = 'cuda'), -1) # 10*8

                a = (a_unit * which_layer_[0]).sum(dim = -1)
                b = (b_unit * which_layer_[1]).sum(dim = -1)
                
                # last layer output
                if i == len(self.layers)-1:
                    x_out = a
                    break

                x_out = 1 - a * b 
                x_in = torch.cat([x_in,x_out],dim=-1)
        else:
            for i in range(len(self.layers)):
                ab = self.layers[i] 
                which_layer = self.which_layers[i]

                if self.descent_layer > 0 and i > 4 :
                    decent = torch.linspace(0.1, 1, which_layer.size(-1), device='cuda')
                    which_layer = torch.abs(which_layer) * decent
                elif self.descent_layer_in and 0 < i <= 4:
                    decent = torch.linspace(1, 0.1, which_layer.size(-1), device='cuda')
                    which_layer = torch.abs(which_layer) * decent 

                which_layer_ = which_layer.argmax(-1)

                start_unit_a, end_unit_a = self.groups[which_layer_[0]], self.groups[which_layer_[0]+1]
                start_unit_b, end_unit_b = self.groups[which_layer_[1]], self.groups[which_layer_[1]+1]
                
                unit = torch.zeros(4,start_unit_a.size(0), dtype=torch.int64, device = 'cuda') # unit_a, inverter_a, unit_b, inverter_b

                for j, (ai, start_a, end_a, bi, start_b, end_b) in enumerate(zip(ab[0], start_unit_a, end_unit_a, ab[1], start_unit_b, end_unit_b)):

                    index_a = torch.abs(ai[start_a:end_a])

                    if hasattr(self, 'reuse_node') and self.reuse_node and which_layer_[0][j].item() <= self.down_layer_num:
                        index_a = index_a * (self.reuse_node_alpha + (1-self.reuse_node_alpha) * self.used_node[start_a:end_a]).unsqueeze(1).repeat(1,2)

                    index_a = index_a.flatten(start_dim=0).argmax()

                    unit[0][j] = index_a//2 + start_a
                    unit[1][j] = index_a % 2
                    # print(f"layer {i}, out {j}, gate {self.groups[i+1]+j}: child1 {unit[0][j]},inv {unit[1][j]}")

                    index_b = torch.abs(bi[start_b:end_b])
                    index_b = index_b.flatten(start_dim=0).argmax()

                    unit[2][j] = index_b//2 + start_b
                    unit[3][j] = index_b % 2
                    # print(f"layer {i}, out {j}, gate {self.groups[i+1]+j}: child2 {unit[2][j]},inv {unit[3][j]}")

                a = (1-unit[1]) * x_in[..., unit[0]] + unit[1] * (1-x_in[..., unit[0]])
                b = (1-unit[3]) * x_in[..., unit[2]] + unit[3] * (1-x_in[..., unit[2]])

                if i == len(self.layers)-1:
                    x_out = a
                    break 

                x_out = 1 - a * b
                x_in = torch.cat([x_in,x_out],dim=-1)

        return x_out

    def find_first_node(self): # find the first node of each output, not used in main.py
        first_node = torch.zeros(self.out_dim, dtype=torch.int64, device = 'cuda')
        first_node_layer = torch.zeros(self.out_dim, dtype=torch.int64, device = 'cuda')
        for node_num in range(self.out_dim):
            which_layer_param = self.which_layers[-1][0][node_num] 

            if self.descent_layer > 0 : # only suit for layer_num > 5
                decent = torch.linspace(0.1, 1,  which_layer_param.size(-1), device='cuda')
                which_layer_param = torch.abs(which_layer_param) * decent

            which_layer = which_layer_param.argmax(-1).item() # layer index
            first_node_layer[node_num]=which_layer
            start_unit, end_unit = self.groups[which_layer], self.groups[which_layer+1] 
            gate_index = torch.abs(self.layers[-1][0][node_num][start_unit:end_unit])
            gate_index = gate_index.flatten(start_dim=0).argmax().item() # unit index

            gate = gate_index//2 + start_unit
            inv = gate_index % 2

            first_node[node_num] = gate

        return first_node, first_node_layer

    def count_connected_node(self):
        gates = torch.zeros(self.layer_sum_list[-1], device = 'cuda') # 0 for not visited, 1 for visited
        self.level = torch.zeros_like(gates, device = 'cuda') # record the level of each node
        self.invs = torch.zeros_like(gates).repeat(2,1).t() # record the number of inverters of each node
        max_level = 0
        for node_num in range(self.out_dim): # depth first search for each output
            which_layer_param = self.which_layers[-1][0][node_num] # choose the first layer of each output

            if self.descent_layer  > 0: # only suit for layer_num > 5
                decent = torch.linspace(0.1, 1, which_layer_param.size(-1), device='cuda')
                which_layer_param = torch.abs(which_layer_param) * decent

            which_layer = which_layer_param.argmax(-1).item() # layer index
            start_unit, end_unit = self.groups[which_layer], self.groups[which_layer+1]
            gate_index = torch.abs(self.layers[-1][0][node_num][start_unit:end_unit])
           
            gate_index = gate_index.flatten(start_dim=0).argmax().item()
            gate = gate_index//2 + start_unit
            inv = gate_index % 2
            # print(f"out {node_num}:{gate},inv {inv}")

            if inv == 1:
                self.invs[self.layer_sum_list[-1]-self.out_dim+node_num,0] = 1
            
            if gate >= self.layer_sum_list[0]:
                gates, level = self.count_connected_node_self(gate, gates) # depth first search
                max_level = max(max_level, level)
        
        # print('total gates = ', gates.sum().item(), 'lev = ', max_level)
        # print(f'total inverters = {self.invs.sum().item()}')
        
        return gates[self.in_dim:].sum().item(),max_level, gates, self.invs.sum().item()

    def count_connected_node_self(self, gate, gates): # depth first search
        if gates[gate] == 1:
            return gates, self.level[gate].item()
        
        layer, node = self.find_gate_position(gate)

        which_layer_a = self.which_layers[layer][0][node]
        if self.descent_layer > 0 and layer > 4 :
            decent = torch.linspace(0.1, 1, which_layer_a.size(-1), device='cuda')
            which_layer_a = torch.abs(which_layer_a) * decent
        elif self.descent_layer_in and 0 < layer <= 4:
            decent = torch.linspace(1, 0.1, which_layer_a.size(-1), device='cuda')
            which_layer_a = torch.abs(which_layer_a) * decent

        which_layer_a = which_layer_a.argmax(-1).item()
        start_unit_a, end_unit_a = self.groups[which_layer_a], self.groups[which_layer_a+1]
        
        a = torch.abs(self.layers[layer][0][node][start_unit_a:end_unit_a])
        a = a.flatten(start_dim=0)

        which_layer_b = self.which_layers[layer][1][node]
        if self.descent_layer > 0 and layer > 4 :
            decent = torch.linspace(0.1, 1, which_layer_b.size(-1), device='cuda')
            which_layer_b = torch.abs(which_layer_b) * decent
        elif self.descent_layer_in and 0 < layer <= 4:
            decent = torch.linspace(1, 0.1, which_layer_b.size(-1), device='cuda')
            which_layer_b = torch.abs(which_layer_b) * decent

        which_layer_b = which_layer_b.argmax(-1).item()
        start_unit_b, end_unit_b = self.groups[which_layer_b], self.groups[which_layer_b+1] 

        b = torch.abs(self.layers[layer][1][node][start_unit_b:end_unit_b])
        b = b.flatten(start_dim=0)
        
        child1 = a.argmax().item() 
        inv1 = child1 % 2
        if inv1 == 1:
            self.invs[gate,0] = 1
        child1 = torch.div(child1, 2, rounding_mode='floor') + start_unit_a
        child2 = b.argmax().item()
        inv2 = child2 % 2
        if inv2 == 1:
            self.invs[gate,1] = 1
        child2 = torch.div(child2, 2, rounding_mode='floor') + start_unit_b

        if child1 >= self.layer_sum_list[0]:
            flag1 = 1
            gates, level1 = self.count_connected_node_self(child1, gates)
        else: 
            gates[child1]=1
            flag1 = 0
        if child2 >= self.layer_sum_list[0]:
            flag2 = 1
            gates, level2 = self.count_connected_node_self(child2, gates)
        else:
            gates[child2]=1
            flag2 = 0

        gates[gate]=1

        # calculate the level of each node
        if flag1 == 0 and flag2 ==0:
            level = 1
        elif flag1 == 1 and flag2 == 1:
            level = max(level1, level2)+1
        elif flag1 == 0 and flag2 == 1:
            level = level2+1
        elif flag1 == 1 and flag2 == 0:
            level = level1 + 1

        self.level[gate] = level

        return gates, level

    def find_gate_position(self, gate):
        assert gate >= self.layer_sum_list[0], 'gate has no input'
        for i in self.layer_sum_list:
            if gate+1 <= i:
                layer = self.layer_sum_list.index(i)-1
                node = gate - self.layer_sum_list[layer]
                break
        return layer, node

########################################################################################################################
