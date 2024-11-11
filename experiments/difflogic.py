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

        ##layers
        self.layer_flatten = torch.nn.Flatten()

        self.layer_list = self.layer_nodenum_list + [self.out_dim]
        self.layers = self.generate_layer(self.layer_list)

        #存了各层的节点数
        self.layer_sum_list = self.layer_list.copy()
        for i in range(len(self.layer_list)):
            self.layer_sum_list[i] = sum(self.layer_list[:i+1])
        self.groups = torch.tensor([0]+self.layer_sum_list)

        self.total_num_neurons = sum(self.layer_nodenum_list[1:])
        
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
                ab = self.layers[i] #10*8
                which_layer = self.which_layers[i] # 2 * out * sum(in) * 2
                
                if hasattr(self, 'pow'):
                    input_layer = torch.pow(ab[:,:,:self.layer_list[0],:],2).flatten(start_dim=-2,end_dim=-1)
                else:
                    input_layer = torch.abs(ab[:,:,:self.layer_list[0],:]).flatten(start_dim=-2,end_dim=-1)
                # input_layer = ab[:,:,:self.layer_list[0],:].flatten(start_dim=-2,end_dim=-1)
                input_layer = torch.nn.functional.gumbel_softmax(input_layer, tau=self.tau, hard=False).view(2, -1, self.layer_list[0], 2) # 2 * out * sum(in) * 2
                

                if hasattr(self, 'pow'):
                    down_body_layer = torch.pow(ab[:,:,self.layer_sum_list[0]:self.layer_sum_list[self.down_layer_num],:],2)
                else:
                    down_body_layer = torch.abs(ab[:,:,self.layer_sum_list[0]:self.layer_sum_list[self.down_layer_num],:]) # 2*out*sum(down_layer)*2

                if hasattr(self, 'reuse_node') and self.reuse_node:
                    down_body_layer = down_body_layer * (self.reuse_node_alpha + (1-self.reuse_node_alpha) * self.used_node[self.layer_sum_list[0]:self.layer_sum_list[0]+down_body_layer.shape[2]]).unsqueeze(1).repeat(1,2)

                down_body_layer = down_body_layer.view(2, ab.size(1), -1, self.down_layer_node, 2)
                down_body_layer = torch.nn.functional.gumbel_softmax(down_body_layer.flatten(start_dim=-2,end_dim=-1), tau=self.tau, hard=False) #2*out*down_layer*(down_node*2)
                down_body_layer = down_body_layer.view(2, ab.size(1),-1,2) #2*out*(down_layer*down_node)
                
                if hasattr(self, 'pow'):
                    up_body_layer = torch.pow(ab[:,:,self.layer_sum_list[self.down_layer_num]:,:],2)
                else:
                    up_body_layer = torch.abs(ab[:,:,self.layer_sum_list[self.down_layer_num]:,:])

                # if hasattr(self, 'reuse_node') and self.reuse_node:
                #     up_body_layer = up_body_layer * (self.reuse_node_alpha + (1-self.reuse_node_alpha) * self.used_node[self.layer_sum_list[self.down_layer_num]:self.layer_sum_list[self.down_layer_num]+up_body_layer.shape[2]]).unsqueeze(1).repeat(1,2)

                up_body_layer = up_body_layer.view(2, ab.size(1), -1, self.up_layer_node, 2) # 2 * out * sum(up) * 2
                up_body_layer = torch.nn.functional.gumbel_softmax(up_body_layer.flatten(start_dim=-2,end_dim=-1), tau=self.tau, hard=False)
                up_body_layer = up_body_layer.view(2, ab.size(1),-1,2)
                
                ab = torch.cat([input_layer, down_body_layer, up_body_layer], dim = -2)

                a_unit = (x_in.unsqueeze(2) * ab[0,:,:,0].t().unsqueeze(0) + ab[0,:,:,1].t().unsqueeze(0) * (1-x_in.unsqueeze(2))).transpose(dim0=-1, dim1=-2)
                b_unit = (x_in.unsqueeze(2) * ab[1,:,:,0].t().unsqueeze(0) + (1-x_in.unsqueeze(2)) * ab[1,:,:,1].t().unsqueeze(0)).transpose(dim0=-1, dim1=-2)


                groups_ = self.layer_list[:i+1]


                if self.descent_layer > 0 and i > 4 :
                    # 生成一个从0.1到1的、长度为1+1的等差数列
                    decent = torch.linspace(0.1, 1, which_layer.size(-1), device='cuda')
                    if hasattr(self, 'pow'):
                        which_layer = torch.pow(which_layer,2) * decent
                    else:
                        which_layer = torch.abs(which_layer) * decent
                elif self.descent_layer_in and 0 < i <= 4:
                    decent = torch.linspace(1, 0.1, which_layer.size(-1), device='cuda')
                    if hasattr(self, 'pow'):
                        which_layer = torch.pow(which_layer,2) * decent
                    else:
                        which_layer = torch.abs(which_layer) * decent

                if hasattr(self, 'out_attention') and self.layer_attention and i == len(self.layers)-1:
                    attenion = self.out_attention
                    which_layer = torch.abs(which_layer) * attenion
                elif hasattr(self, 'layer_attention') and self.layer_attention and i > 0:
                    attenion = self.attention_list[i]
                    which_layer = torch.abs(which_layer) * attenion

                which_layer = torch.nn.functional.gumbel_softmax(which_layer, tau=self.tau, hard=False) 
                which_layer_ = torch.repeat_interleave(which_layer, torch.tensor(groups_ ,device = 'cuda'), -1) # 10*8

                a = (a_unit * which_layer_[0]).sum(dim = -1)
                b = (b_unit * which_layer_[1]).sum(dim = -1)
                
                # 最后一层，直接输出，不经过与非门
                if i == len(self.layers)-1:
                    x_out = a
                    break

                x_out = 1 - a * b # 4 * 10
                x_in = torch.cat([x_in,x_out],dim=-1)
        else:
            for i in range(len(self.layers)):
                ab = self.layers[i] 
                which_layer = self.which_layers[i]
                
                if hasattr(self, 'out_attention') and self.layer_attention and i == len(self.layers)-1:
                    attenion = self.out_attention
                    which_layer = torch.abs(which_layer) * attenion
                elif hasattr(self, 'layer_attention') and self.layer_attention and i > 0:
                    attenion = self.attention_list[i]
                    which_layer = torch.abs(which_layer) * attenion

                if self.descent_layer > 0 and i > 4 :
                # 生成一个从0.1到1的、长度为1+1的等差数列
                    decent = torch.linspace(0.1, 1, which_layer.size(-1), device='cuda')
                    if hasattr(self, 'pow'):
                        which_layer = torch.pow(which_layer,2) * decent
                    else:
                        which_layer = torch.abs(which_layer) * decent
                elif self.descent_layer_in and 0 < i <= 4:
                    decent = torch.linspace(1, 0.1, which_layer.size(-1), device='cuda')
                    if hasattr(self, 'pow'):
                        which_layer = torch.pow(which_layer,2) * decent
                    else:
                        which_layer = torch.abs(which_layer) * decent 

                which_layer_ = which_layer.argmax(-1)

                start_unit_a, end_unit_a = self.groups[which_layer_[0]], self.groups[which_layer_[0]+1]
                start_unit_b, end_unit_b = self.groups[which_layer_[1]], self.groups[which_layer_[1]+1]
                
                unit = torch.zeros(4,start_unit_a.size(0), dtype=torch.int64, device = 'cuda') # 分别存unit_a，inverter_a unit_b inverter_b

                for j, (ai, start_a, end_a, bi, start_b, end_b) in enumerate(zip(ab[0], start_unit_a, end_unit_a, ab[1], start_unit_b, end_unit_b)):

                    if hasattr(self, 'pow'):
                        index_a = torch.pow(ai[start_a:end_a],2)
                    else:
                        index_a = torch.abs(ai[start_a:end_a])

                    if hasattr(self, 'reuse_node') and self.reuse_node and which_layer_[0][j].item() <= self.down_layer_num:
                        index_a = index_a * (self.reuse_node_alpha + (1-self.reuse_node_alpha) * self.used_node[start_a:end_a]).unsqueeze(1).repeat(1,2) # 不连接的节点logits减半

                    index_a = index_a.flatten(start_dim=0).argmax()

                    unit[0][j] = index_a//2 + start_a
                    unit[1][j] = index_a % 2
                    # print(f"layer {i}, out {j}, gate {self.groups[i+1]+j}: child1 {unit[0][j]},inv {unit[1][j]}")

                    if hasattr(self, 'pow'):
                        index_b = torch.pow(bi[start_b:end_b],2)
                    else:
                        index_b = torch.abs(bi[start_b:end_b])
                    # index_b = bi[start_b:end_b]

                    if hasattr(self, 'reuse_node') and self.reuse_node and which_layer_[1][j].item() <= self.down_layer_num:
                        index_b = index_b * (self.reuse_node_alpha + (1-self.reuse_node_alpha) * self.used_node[start_b:end_b]).unsqueeze(1).repeat(1,2)
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

    # 找到每个输出位连接的第一个节点的编号
    def find_first_node(self):
        first_node = torch.zeros(self.out_dim, dtype=torch.int64, device = 'cuda')
        first_node_layer = torch.zeros(self.out_dim, dtype=torch.int64, device = 'cuda')
        for node_num in range(self.out_dim):
            which_layer_param = self.which_layers[-1][0][node_num] # 选出层参数 1*60

            if self.descent_layer  > 0 : #未考虑最后一层小于5的情况
                # 生成一个从0.1到1的、长度为1+1的等差数列
                decent = torch.linspace(0.1, 1,  which_layer_param.size(-1), device='cuda')
                if hasattr(self, 'pow'):
                    which_layer_param = torch.pow(which_layer_param,2) * decent
                else:
                    which_layer_param = torch.abs(which_layer_param) * decent
                # which_layer_param = which_layer_.argmax(-1).item()

            if hasattr(self, 'out_attention') and self.layer_attention:
                attenion = self.out_attention[node_num]
                which_layer_param = torch.abs(which_layer_param) * attenion
            elif hasattr(self, 'layer_attention') and self.layer_attention > 0:
                attenion = self.attention_list[-1]
                which_layer_param = torch.abs(which_layer_param) * attenion

            which_layer = which_layer_param.argmax(-1).item()# 选出要找哪一层
            first_node_layer[node_num]=which_layer
            start_unit, end_unit = self.groups[which_layer], self.groups[which_layer+1] #挑选出对应的node编号
            # gate_index = torch.abs(self.layers[-1][0][node_num][start_unit:end_unit])
            if hasattr(self, 'pow'):
                gate_index = torch.pow(self.layers[-1][0][node_num][start_unit:end_unit],2)
            else:
                gate_index = torch.abs(self.layers[-1][0][node_num][start_unit:end_unit])
           
            
            if hasattr(self, 'reuse_node') and self.reuse_node and which_layer <= self.down_layer_num:
                gate_index = gate_index * (self.reuse_node_alpha + (1-self.reuse_node_alpha) * self.used_node[start_unit:end_unit]).unsqueeze(1).repeat(1,2)
            
            gate_index = gate_index.flatten(start_dim=0).argmax().item() #挑选出对应的unit编号

            gate = gate_index//2 + start_unit
            inv = gate_index % 2

            first_node[node_num] = gate
        return first_node, first_node_layer

    def count_connected_node(self):
        gates = torch.zeros(self.layer_sum_list[-1], device = 'cuda') #统计节点数的tensor，0表示这个节点没走过，1表示走过
        self.level = torch.zeros_like(gates, device = 'cuda') #统计每个节点的level
        self.invs = torch.zeros_like(gates).repeat(2,1).t() #统计非门inverter
        max_level = 0
        for node_num in range(self.out_dim): # 对网络深度优先搜索，走过的节点置1
            # gate = self.layer_sum_list[-1]-1-node_num
            which_layer_param = self.which_layers[-1][0][node_num] # 选出层参数 1*60

            if self.descent_layer  > 0: #未考虑网络深度小于5的情况
                # 生成一个从0.1到1的、长度为1+1的等差数列
                decent = torch.linspace(0.1, 1, which_layer_param.size(-1), device='cuda')
                if hasattr(self, 'pow'):
                    which_layer_param = torch.pow(which_layer_param,2) * decent
                else:
                    which_layer_param = torch.abs(which_layer_param) * decent
                # which_layer = which_layer_.argmax(-1).item()
            # else:
            if hasattr(self, 'out_attention') and self.layer_attention:
                attenion = self.out_attention[node_num]
                which_layer_param = torch.abs(which_layer_param) * attenion
            elif hasattr(self, 'layer_attention') and self.layer_attention:
                attenion = self.attention_list[-1]
                which_layer_param = torch.abs(which_layer_param) * attenion

            which_layer = which_layer_param.argmax(-1).item()
            # which_layer = which_layer_param.argmax(-1).item() # 选出要找哪一层
            start_unit, end_unit = self.groups[which_layer], self.groups[which_layer+1] #挑选出对应的node编号
            # gate_index = torch.abs(self.layers[-1][0][node_num][start_unit:end_unit]).flatten(start_dim=0).argmax().item() #挑选出对应的unit编号
            # gate_index = torch.abs(self.layers[-1][0][node_num][start_unit:end_unit])
            if hasattr(self, 'pow'):
                gate_index = torch.pow(self.layers[-1][0][node_num][start_unit:end_unit],2)
            else:
                gate_index = torch.abs(self.layers[-1][0][node_num][start_unit:end_unit])
           

            if hasattr(self, 'reuse_node') and self.reuse_node and which_layer <= self.down_layer_num:
                gate_index = gate_index * (self.reuse_node_alpha + (1-self.reuse_node_alpha) * self.used_node[start_unit:end_unit]).unsqueeze(1).repeat(1,2)
            
            gate_index = gate_index.flatten(start_dim=0).argmax().item()
            gate = gate_index//2 + start_unit
            inv = gate_index % 2
            # print(f"out {node_num}:{gate},inv {inv}")

            if inv == 1:
                self.invs[self.layer_sum_list[-1]-self.out_dim+node_num,0] = 1
            
            if gate >= self.layer_sum_list[0]:
                gates, level = self.count_connected_node_self(gate, gates)
                max_level = max(max_level, level)
        
        # print('total gates = ', gates.sum().item(), 'lev = ', max_level)
        # print(f'total inverters = {self.invs.sum().item()}')
        
        return gates[self.in_dim:].sum().item(),max_level, gates, self.invs.sum().item()

    def count_connected_node_self(self, gate, gates): # 递归搜索函数
        if gates[gate] == 1:
            return gates, self.level[gate].item()
        # else:
        
        layer, node = self.find_gate_position(gate)

        which_layer_a = self.which_layers[layer][0][node]
        if self.descent_layer > 0 and layer > 4 :
            # 生成一个从0.1到1的、长度为1+1的等差数列
            decent = torch.linspace(0.1, 1, which_layer_a.size(-1), device='cuda')
            if hasattr(self, 'pow'):
                which_layer_a = torch.pow(which_layer_a,2) * decent
            else:
                which_layer_a = torch.abs(which_layer_a) * decent
        elif self.descent_layer_in and 0 < layer <= 4:
            decent = torch.linspace(1, 0.1, which_layer_a.size(-1), device='cuda')
            if hasattr(self, 'pow'):
                which_layer_a = torch.pow(which_layer_a,2) * decent
            else:
                which_layer_a = torch.abs(which_layer_a) * decent
        if hasattr(self, 'layer_attention') and self.layer_attention and layer > 0:
            attenion = self.attention_list[layer]
            which_layer_a = torch.abs(which_layer_a) * attenion
        which_layer_a = which_layer_a.argmax(-1).item()
        start_unit_a, end_unit_a = self.groups[which_layer_a], self.groups[which_layer_a+1] #挑选出对应的node编号
        # a = torch.abs(self.layers[layer][0][node][start_unit_a:end_unit_a]).flatten(start_dim=0) #挑选出对应的unit编号

        # a = torch.abs(self.layers[layer][0][node][start_unit_a:end_unit_a])
        if hasattr(self, 'pow'):
            a = torch.pow(self.layers[layer][0][node][start_unit_a:end_unit_a],2)
        else:
            a = torch.abs(self.layers[layer][0][node][start_unit_a:end_unit_a])
            
        if hasattr(self, 'reuse_node') and self.reuse_node and which_layer_a <= self.down_layer_num:
            a = a * (self.reuse_node_alpha + (1-self.reuse_node_alpha) * self.used_node[start_unit_a:end_unit_a]).unsqueeze(1).repeat(1,2)
        
        a = a.flatten(start_dim=0)

        which_layer_b = self.which_layers[layer][1][node]
        if self.descent_layer > 0 and layer > 4 :
            # 生成一个从0.1到1的、长度为1+1的等差数列
            decent = torch.linspace(0.1, 1, which_layer_b.size(-1), device='cuda')
            if hasattr(self, 'pow'):
                which_layer_b = torch.pow(which_layer_b,2) * decent
            else:
                which_layer_b = torch.abs(which_layer_b) * decent
        elif self.descent_layer_in and 0 < layer <= 4:
            decent = torch.linspace(1, 0.1, which_layer_b.size(-1), device='cuda')
            if hasattr(self, 'pow'):
                which_layer_b = torch.pow(which_layer_b,2) * decent
            else:
                which_layer_b = torch.abs(which_layer_b) * decent
        if hasattr(self, 'layer_attention') and self.layer_attention and layer > 0:
            attenion = self.attention_list[layer]
            which_layer_b = torch.abs(which_layer_b) * attenion
        which_layer_b = which_layer_b.argmax(-1).item()

        start_unit_b, end_unit_b = self.groups[which_layer_b], self.groups[which_layer_b+1] #挑选出对应的node编号
        # b = torch.abs(self.layers[layer][1][node][start_unit_b:end_unit_b]).flatten(start_dim=0) #挑选出对应的unit编号
        # b = torch.abs(self.layers[layer][1][node][start_unit_b:end_unit_b])
        if hasattr(self, 'pow'):
            b = torch.pow(self.layers[layer][1][node][start_unit_b:end_unit_b],2)
        else:
            b = torch.abs(self.layers[layer][1][node][start_unit_b:end_unit_b])
            
        if hasattr(self, 'reuse_node') and self.reuse_node and which_layer_b <= self.down_layer_num:
            b = b * (self.reuse_node_alpha + (1-self.reuse_node_alpha) * self.used_node[start_unit_b:end_unit_b]).unsqueeze(1).repeat(1,2)
        
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

        # # 保存网络结构
        # if gates[gate] == 0:
        #     # with open("/difflogic/exp6/network_visualization/600.txt", "a") as file:
        #     #     file.write(f"{gate}({layer},{node}) -> {child1} + {child2}\n")
        #     print(f"{gate}({layer},{node}) -> {child1} + {child2} , inv1 = {inv1}, inv2 = {inv2}")

        
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

        # if gates[gate] == 0:
        #     with open("/difflogic/exp6/network_visualization/600.txt", "a") as file:
        #         file.write(f"{gate}({layer},{node}) -> {child1} + {child2}\n")
            # print(f"{gate}({layer},{node}) -> {child1} + {child2} , inv1 = {inv1}, inv2 = {inv2}")
        gates[gate]=1

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
