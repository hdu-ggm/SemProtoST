import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.UniTraffic.arch.PatternBank import PatternRelationBank
from baselines.UniTraffic.arch.mlp import MultiLayerPerceptron

class LargeSTMetaModel(nn.Module):
    """
    Inductive Spatio-Temporal Model for LargeST.
    Processes traffic flow and metadata to generate latent tokens and predictions.
    """

    def __init__(self, **model_args):
        super().__init__()

        # 1. 基础配置
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]

        self.district_num = model_args["district"]
        self.district_dim = model_args["district_dim"]
        self.county_num = model_args['county']
        self.county_dim = model_args['county_dim']
        self.fwy_num = model_args['fwy']
        self.fwy_dim = model_args['fwy_dim']
        self.direction_num = model_args['direction']
        self.direction_dim = model_args['direction_dim']
        self.number_size = model_args['number_size']
        self.number_dim  = model_args['number_dim']
        self.num_s_prototypes = model_args['num_s_prototypes']
        self.num_t_prototypes = model_args['num_t_prototypes']
        # 2. 类别型元数据处理 (Categorical Metadata) 根据 District, County, Fwy, Type, Direction 的类别数量创建 Embedding
        if self.district_num != 1:
            self.district = nn.Parameter(torch.empty(self.district_num, self.district_dim))
            nn.init.xavier_uniform_(self.district)
        else:
            self.district = None
        if self.county_num != 1:
            self.county = nn.Parameter(torch.empty(self.county_num, self.county_dim))
            nn.init.xavier_uniform_(self.county)
        else:
            self.county = None
        if self.fwy_num != 1:
            self.fwy = nn.Parameter(torch.empty(self.fwy_num, self.fwy_dim))
            nn.init.xavier_uniform_(self.fwy)
        else:
            self.fwy = None
        if self.direction_num != 1:
            self.direction = nn.Parameter(torch.empty(self.direction_num, self.direction_dim))
            nn.init.xavier_uniform_(self.direction)
        else:
            self.direction = None

        # 3. 数值型元数据处理 (Lat, Lng, Lanes)
        self.num_projector = nn.Sequential(
            nn.Linear(self.number_size, self.number_dim),
            nn.LeakyReLU(),
            nn.Linear(self.number_dim, self.number_dim)
        )

        # self.node_dim = self.fwy_dim + self.direction_dim + self.number_dim
        self.node_dim = int(self.district_num != 1) * self.district_dim + int(self.county_num != 1) * self.county_dim + int(self.fwy_num != 1) * self.fwy_dim + int(self.direction_num != 1) * self.direction_dim + self.number_dim
        self.spatial_bank = PatternRelationBank(num_prototypes=self.num_s_prototypes, embed_dim=self.node_dim)
        self.temporal_bank = PatternRelationBank(num_prototypes=self.num_t_prototypes, embed_dim=self.temp_dim_tid + self.temp_dim_diw)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_time_in_day) + \
            self.temp_dim_diw*int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, history_data, future_data, meta_cat, meta_num, batch_seen=None, epoch=None, train=True, **kwargs):
        """
        history_data: [B, L, N, 1]
        meta_cat: [B, N, 5] (District, County, Fwy, Type, Direction)
        meta_num: [B, N, 3] (Lat, Lng, Lanes)
        """
        # prepare data
        input_data = history_data[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[
                (d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        # --- B. 空间/元数据特征提取 ---
        # 处理类别型 (Categorical)
        valid_list = [
            (self.district_num, self.district, 0),
            (self.county_num,   self.county,   1),
            (self.fwy_num,      self.fwy,      2),
            (self.direction_num,self.direction, 4)
        ]
        valid_feats = [feat[meta_cat[..., i]] for num, feat, i in valid_list if num > 1]

        cat_feats = torch.cat(valid_feats, dim=-1).transpose(1, 2).unsqueeze(-1) if valid_feats else None
        # 处理数值型 (Numerical)
        num_feats = self.num_projector(meta_num).transpose(1,2).unsqueeze(-1)  # [B, D, N, 1]

        spa_enhanced_token, spa_attn, loss_ortho_s = self.spatial_bank(torch.cat([cat_feats, num_feats], dim=1))

        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        tem_enhanced_token, tem_attn, loss_ortho_t = self.temporal_bank(torch.cat(tem_emb, dim=1))

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + [spa_enhanced_token] + [tem_enhanced_token], dim=1)

        # encoding
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden)

        return prediction, loss_ortho_s+loss_ortho_t, spa_attn, torch.cat([meta_cat, meta_num], dim=-1)

        # return prediction, loss_ortho_s+loss_ortho_t, spa_attn, torch.cat([cat_feats, num_feats], dim=1), spa_enhanced_token

