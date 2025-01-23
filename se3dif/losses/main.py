from .denoising_loss import ProjectedSE3DenoisingLoss, SE3DenoisingLoss
from .sdf_loss import SDFLoss


def get_losses(args):
    losses = args['Losses']

    loss_fns = {}
    if 'sdf_loss' in losses:
        loss_fns['sdf'] = SDFLoss()
    if 'projected_denoising_loss' in losses:
        loss_fns['denoise'] = ProjectedSE3DenoisingLoss()
    if 'denoising_loss' in losses:
        loss_fns['denoise'] = SE3DenoisingLoss()

    loss_dict = LossDictionary(loss_dict=loss_fns)
    return loss_dict


class LossDictionary():

    def __init__(self, loss_dict):
        self.fields = loss_dict.keys()
        self.loss_dict = loss_dict

    def loss_fn(self, model, model_input, ground_truth, val=False):
        losses = {}
        infos = {}
        
        # set the visual context for the model
        c = model_input['visual_context']
        model.set_latent(c)
        
        for field in self.fields:
            loss_fn_k = self.loss_dict[field]
            loss, info = loss_fn_k(model, model_input, ground_truth, val)
            losses = {**losses, **loss}
            infos = {**infos, **info}

        return losses, infos