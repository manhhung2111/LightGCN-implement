import torch as t
import torch.nn.functional as F

def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
	pos_preds = (anc_embeds * pos_embeds).sum(-1)
	neg_preds = (anc_embeds * neg_embeds).sum(-1)
	return t.sum(F.softplus(neg_preds - pos_preds))


def reg_pick_embeds(embeds_list):
	reg_loss = 0
	for embeds in embeds_list:
		reg_loss += embeds.square().sum()
	return reg_loss


def reg_params(model):
	reg_loss = 0
	for W in model.parameters():
		reg_loss += W.norm(2).square()
	return reg_loss
