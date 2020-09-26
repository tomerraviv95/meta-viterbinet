from python_code.decoders.WCVA.wcva_decoder import WCVADecoder
from python_code.trainers.trainer import Trainer
from python_code.utils.tail_biting_utils import calculate_starting_state_for_tbcc
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WCVATrainer(Trainer):
    """
    Trainer for the WCVA model.
    """

    def __init__(self, config_path=None, **kwargs):
        self.replications = None
        super().__init__(config_path, **kwargs)

    def __name__(self):
        return f'{self.replications}-rep WCVA'

    def load_detector(self):
        """
        Loads the WCVA decoder
        """
        self.decoder = WCVADecoder(det_length=self.det_length,
                                   replications=self.replications,
                                   n_states=self.n_states,
                                   clipping_val=self.clipping_val,
                                   code_length=self.code_length,
                                   code_gm=self.code_gm_inner)

        if self.load_from_checkpoint:
            self.load_last_checkpoint()

    def calc_loss(self, soft_estimation: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        start_states = calculate_starting_state_for_tbcc(self.n_states, labels)
        # the cross entropy loss, see paper for more details
        loss = self.criterion(input=soft_estimation[:, -self.n_states:],
                              target=start_states)
        return loss


if __name__ == '__main__':
    dec = WCVATrainer()
    dec.train()
