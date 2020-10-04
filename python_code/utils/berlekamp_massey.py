import sys
import numpy as np
import os

import torch

sys.path.insert(0, os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BerlekampMasseyHardDecoder:
    def __init__(self, bits_num, t):
        super(BerlekampMasseyHardDecoder, self).__init__()
        self.bits_num = bits_num
        self.t = t
        self.initialize_poly()
        self.antilog_table, self.log_table = self.create_lookup_tables()

    def initialize_poly(self):
        if self.bits_num == 63:
            self.poly = [1, 1, 0, 0, 0, 0, 1]
        elif self.bits_num == 255:
            self.poly = [1, 0, 0, 0, 1, 1, 1, 0, 1]
        else:
            raise Warning('poly is unknown for this code')

    def create_lookup_tables(self):
        '''
        Lookup tables:
        antilog_table: exponent of alpha to decimal value of binary
        log_table: decimal value of binary to exponent of alpha
        :return:
        '''

        antilog_table = np.zeros(self.bits_num, dtype=int)
        log_table = np.zeros(self.bits_num + 1, dtype=int)
        m = int(np.log2(self.bits_num + 1))
        antilog_table[0] = 1
        log_table[0] = -1
        for i in range(1, self.bits_num):
            antilog_table[i] = antilog_table[i - 1] << 1
            if antilog_table[i] >= 2 ** m:
                antilog_table[i] ^= int("".join(str(x) for x in self.poly[::-1]), 2)

            log_table[antilog_table[i]] = i
        return torch.Tensor(antilog_table).int().to(device=device), \
               torch.Tensor(log_table).int().to(device=device)

    def array_map(self, to_map_array, mapping_values):
        if len(to_map_array.shape) == 1:
            result = np.array([mapping_values[x].cpu().numpy().item() for x in to_map_array])
        else:
            result = np.array([[mapping_values[y].cpu().numpy().item() for y in x] for x in to_map_array])
        return result

    def forward(self, hard_decoded_xs):
        outputs = hard_decoded_xs.clone()
        t2 = 2 * self.t
        s, syn_error = self.compute_syndrome(hard_decoded_xs, t2)
        errors_num_array = torch.zeros(hard_decoded_xs.shape[0], dtype=torch.int)

        if torch.sum(syn_error) == 0:
            return outputs, errors_num_array

        d, elp, l, u_lu = self.initialize(s, t2)
        unsatisfied = torch.nonzero(syn_error)[:, 0].long()

        for u in range(1, t2 + 1):
            l[unsatisfied], elp[unsatisfied] = self.iterate(d[unsatisfied], elp[unsatisfied], l[unsatisfied],
                                                            u, u_lu[unsatisfied])
            u_lu[unsatisfied, u + 1] = u - l[unsatisfied, u + 1]

            if u < t2:
                d[unsatisfied] = self.compute_next_discrepancy(d[unsatisfied], elp[unsatisfied],
                                                               l[unsatisfied],
                                                               [s[i] for i in unsatisfied.cpu().numpy()], u)
                unsatisfied = unsatisfied[l[unsatisfied, u + 1] <= self.t]
            # if l[u + 1] > self.t:
            #     break

        u = u + 1
        l_u_ind = self.init_l_u_ind(l[unsatisfied], u)
        # l_u_ind = torch.LongTensor([x.cpu().numpy() for x in l_u_ind if x[0].item() in unsatisfied])

        outputs[unsatisfied] = self.chiens_search(hard_decoded_xs[unsatisfied],
                                                  elp[unsatisfied],
                                                  l[unsatisfied], self.t, t2, u,
                                                  l_u_ind)
        return outputs.float()

    def init_l_u_ind(self, l, u):
        '''
        return indices of (i,u[i],l[i,u[i]]+1)
        :param l:
        :param u:
        :return:
        '''
        if type(u) == int:
            u = torch.ones(l.shape[0]).long() * u

        total_list = [torch.cartesian_prod(torch.Tensor([x]).int(), torch.Tensor([u[x]]).int(),
                                           torch.arange(l[x, u[x]] + 1).int()) for x in
                      range(l.shape[0])]
        flat_list = []
        for sublist in total_list:
            for item in sublist:
                flat_list.append(item)
        up_to_l_u_indices = torch.stack(flat_list, dim=0).long()
        return up_to_l_u_indices

    def compute_syndrome(self, hard_decoded_xs, t2):
        '''
        computes the syndrome as in 3.17 in p.56
        :param hard_decoded_xs: the received words
        :param t2: 2*t
        :return: syndrome and a flag indicating whether an error occured
        '''
        synd_equations = torch.IntTensor(
            np.mod(np.arange(1, t2 + 1).reshape(-1, 1) * np.arange(0, self.bits_num), self.bits_num)).to(device=device)
        received_synd_array = [synd_equations[:, hard_decoded_x.bool()] for hard_decoded_x in hard_decoded_xs]
        synd_values_array = [torch.IntTensor(
            np.bitwise_xor.reduce(self.array_map(received_synd.int(), self.antilog_table), axis=1)).to(device) for
                             received_synd in received_synd_array]
        syn_error = torch.BoolTensor([torch.any(synd_values > 0).item() for synd_values in synd_values_array]).to(
            device=device)
        synd_exp_array = [torch.IntTensor(self.array_map(synd_values, self.log_table)).to(device=device) for synd_values
                          in
                          synd_values_array]
        s = [torch.cat([torch.zeros(1, dtype=torch.int).to(device), synd_exp]) for synd_exp in synd_exp_array]
        return s, syn_error

    def initialize(self, s, t2):
        d = torch.zeros([len(s), t2 + 1]).int().to(device=device)
        u_lu = torch.zeros([len(s), t2 + 10]).int().to(device=device)
        l = torch.zeros([len(s), t2 + 5]).int().to(device=device)
        elp = torch.zeros([len(s), t2 + 5, t2 + 5]).int().to(device=device)
        d[:, 0] = 0
        d[:, 1] = torch.IntTensor([s[i][1] for i in range(len(s))]).to(device=device)
        elp[:, 0, 0] = 0
        elp[:, 1, 0] = 1
        elp[:, 0, 1:t2] = -1
        l[:, 0] = 0
        l[:, 1] = 0
        u_lu[:, 0] = -1
        u_lu[:, 1] = 0
        return d, elp, l, u_lu

    def iterate(self, d, elp, l, u, u_lu):

        trivial_update_flags = d[:, u] == -1
        if trivial_update_flags.sum() >= 1:
            up_to_l_u_indices = self.init_l_u_ind(l[trivial_update_flags], u)
            l[trivial_update_flags], elp[trivial_update_flags] = self.d_zero_update(elp[trivial_update_flags],
                                                                                    l[trivial_update_flags], u,
                                                                                    up_to_l_u_indices)
        non_trivial_update_flags = d[:, u] != -1
        if non_trivial_update_flags.sum() >= 1:
            ## not parallelerized
            q = self.find_maximal_u_lu_ind(d[non_trivial_update_flags], u,
                                           u_lu[non_trivial_update_flags])
            l[non_trivial_update_flags, u + 1] = torch.max(l[non_trivial_update_flags, u],
                                                           l[non_trivial_update_flags, q] + u - q.int())
            ## not parallelerized
            elp[non_trivial_update_flags] = self.update_elp(d[non_trivial_update_flags], elp[non_trivial_update_flags],
                                                            l[non_trivial_update_flags], q, u)
        return l, elp

    def d_zero_update(self, elp, l, u, up_to_l_u_indices):

        l[:, u + 1] = l[:, u]
        elp[up_to_l_u_indices[:, 0], up_to_l_u_indices[:, 1] + 1, up_to_l_u_indices[:, 2]] = \
            elp[up_to_l_u_indices[:, 0], up_to_l_u_indices[:, 1], up_to_l_u_indices[:, 2]]
        elp[up_to_l_u_indices[:, 0], up_to_l_u_indices[:, 1], up_to_l_u_indices[:, 2]] = \
            torch.IntTensor(
                self.array_map(elp[up_to_l_u_indices[:, 0], up_to_l_u_indices[:, 1], up_to_l_u_indices[:, 2]],
                               self.log_table)).to(
                device=device)

        return l, elp

    def find_maximal_u_lu_ind(self, d, u, u_lu):
        q = torch.zeros(d.shape[0]).long().to(device=device)
        for i in range(d.shape[0]):
            non_trivial_d_vals_indices = torch.nonzero(d[i, :u] != -1)[:, 0]
            highest_difference_loc = torch.argmax(u_lu[i, non_trivial_d_vals_indices])
            q[i] = non_trivial_d_vals_indices[highest_difference_loc].item()
        return q

    def update_elp(self, d, elp, l, q, u):
        lower_ind1, upper_ind1 = u - q, u - q.int() + l[torch.arange(l.shape[0]), q] + 1
        upper_ind2 = l[torch.arange(l.shape[0]), q.long()] + 1
        for i in range(d.shape[0]):
            existing_indices = (elp[i, q[i], :l[i, q[i]] + 1] != -1).bool()
            ind1 = torch.arange(lower_ind1[i], upper_ind1[i])[existing_indices]
            ind2 = torch.arange(upper_ind2[i])[existing_indices]
            elp[i, u + 1, ind1] = torch.IntTensor(
                self.array_map((d[i, u] - d[i, q[i]] + elp[i, q[i], ind2]) % self.bits_num,
                               self.antilog_table)).to(device)
            elp[i, u + 1, :l[i, u] + 1] ^= elp[i, u, :l[i, u] + 1]
            elp[i, u, :l[i, u] + 1] = torch.IntTensor(self.array_map(elp[i, u, :l[i, u] + 1], self.log_table)).to(
                device)

        return elp

    def compute_next_discrepancy(self, d, elp, l, s, u):
        for i in range(len(s)):

            if s[i][u + 1] != -1:
                d[i, u + 1] = self.antilog_table[s[i][u + 1]]
            else:
                d[i, u + 1] = 0

            syndrome_non_zeros = (s[i][u - l[i, u + 1] + 1:u + 1] != -1)
            inv_idx = torch.arange(syndrome_non_zeros.size(0) - 1, -1, -1).long().to(device)
            sigma_non_zeros = elp[i, u + 1, 1:l[i, u + 1] + 1] != 0
            sigma_exp = torch.IntTensor(self.array_map(elp[i, u + 1, 1:l[i, u + 1] + 1], self.log_table)).to(
                device=device)
            syndrome_non_zeros = syndrome_non_zeros[inv_idx].bool()
            syndrome_exp = s[i][u - l[i, u + 1] + 1:u + 1]
            syndrome_exp = syndrome_exp[inv_idx]
            joint_mask = syndrome_non_zeros & sigma_non_zeros
            try:
                d[i, u + 1] = d[i, u + 1] ^ torch.IntTensor([np.bitwise_xor.reduce(
                    self.array_map((sigma_exp[joint_mask] + syndrome_exp[joint_mask]) % self.bits_num,
                                   self.antilog_table))]).cuda()
                d[i, u + 1] = self.log_table[d[i, u + 1]]
            except Exception:
                d[i, u + 1] = self.log_table[d[i, u + 1]]
        return d

    def chiens_search(self, hard_decoded_x, elp, l, t, t2, u, l_u_ind):
        reg = torch.zeros([hard_decoded_x.shape[0], t2 + 1]).int().to(device=device)
        root = torch.zeros([hard_decoded_x.shape[0], t2 + 1]).int().to(device=device)
        loc = torch.zeros([hard_decoded_x.shape[0], t2 + 1]).long().to(device=device)
        errors_num = torch.zeros(hard_decoded_x.shape[0]).int()
        recd = hard_decoded_x.clone()
        self.initialize_gamma_coef(elp, reg, l_u_ind)
        counts = np.zeros(hard_decoded_x.shape[0])
        for i in range(hard_decoded_x.shape[0]):
            if l[i, u] <= t:
                counts[i] = self.run_brute_force(l[i], loc[i], reg[i], root[i], u)
                self.fix_errors(counts[i], l[i], loc[i], recd[i], u)
            errors_num[i] = l[i, u]
        return recd

    def fix_errors(self, count, l, loc, recd, u):
        if count == l[u]:
            recd[loc[:l[u]]] = recd[loc[:l[u]]] ^ torch.Tensor([1]).int().to(device=device)

    def run_brute_force(self, l, loc, reg, root, u):
        total_chien_matrix = np.matmul(np.arange(self.bits_num).reshape(-1, 1),
                                       np.arange(1, l[u].item() + 1).reshape(1, -1))
        total_mat = self.array_map(
            (torch.IntTensor(total_chien_matrix).to(device) + reg[1:l[u] + 1].reshape(1, -1)) % self.bits_num,
            self.antilog_table)
        mask = (reg[1:l[u] + 1] != -1).cpu().numpy()
        to_reduce_mat = np.hstack([np.ones(self.bits_num, dtype=np.int64).reshape(-1, 1), total_mat[:, mask]])
        q_vals = np.bitwise_xor.reduce(to_reduce_mat, axis=1)
        roots_powers = torch.LongTensor(np.nonzero(q_vals == 0)[0]).to(device)
        count = roots_powers.shape[0]
        root[:count] = roots_powers
        loc[:count] = (self.bits_num - roots_powers) % self.bits_num
        return count

    def initialize_gamma_coef(self, elp, reg, l_u_ind):
        reg[l_u_ind[:, 0], l_u_ind[:, 2]] = torch.IntTensor(
            self.array_map(elp[l_u_ind[:, 0], l_u_ind[:, 1], l_u_ind[:, 2]], self.log_table)).to(
            device)


if __name__ == "__main__":
    dec = BerlekampMasseyHardDecoder(63, 5)
    tx = [0, 0, 1, 1, 1, 1, 1, 0, 0, 1,
          0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
          0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
          1, 1, 1, 1, 0, 1, 0, 0, 0, 0,
          1, 1, 1, 1, 0, 1, 0, 0, 0, 0,
          1, 0, 1, 1, 0, 0, 1, 1, 0, 1,
          0, 0, 0]
    rx = [0, 0, 1, 1, 1, 1, 1, 0, 0, 1,
          0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
          1, 1, 0, 0, 0, 1, 1, 0, 0, 1,
          1, 1, 1, 1, 0, 1, 0, 0, 0, 0,
          1, 1, 1, 1, 0, 1, 0, 0, 1, 0,
          1, 0, 1, 1, 0, 0, 1, 1, 0, 1,
          0, 1, 0]

    hard_decoded_x = torch.Tensor([rx]).int().to(device=device)
    print("received word: " + str(hard_decoded_x))

    corrected_word = dec.forward(hard_decoded_x)
    print("corrected word: " + str(corrected_word))

    flips_num = torch.sum(torch.abs(corrected_word - hard_decoded_x))
    print("flips: " + str(flips_num))

    torch_target = torch.IntTensor(tx).cuda()
    errors_num = torch.sum(torch.abs(corrected_word - torch_target))
    print("errors: " + str(errors_num))
